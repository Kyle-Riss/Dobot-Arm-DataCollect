"""
VLA 데이터 자동 수집 프로그램 (Teaching Log 재생 + 카메라 녹화)

수동 티칭으로 미리 만들어둔 로그를 자동 재생하면서 카메라로 촬영합니다.

워크플로우:
1. logs/ 폴더에서 teaching log 선택
2. 물체를 초기 위치에 세팅
3. 사람은 카메라 시야 밖으로 이동
4. 로봇이 자동으로 로그를 재생하면서 각 포인트에서 촬영
5. VLA 데이터셋 생성 (640x480 이미지 + 로봇 상태)

장점:
- 한 번 teaching하면 여러 번 재활용 가능
- 사람 없는 깨끗한 데이터
- 정확한 타임스텝 동기화 (로봇 정지 후 촬영)
"""

import sys
import os
import cv2
import numpy as np
from ctypes import *
import time
import csv
from datetime import datetime

# HIKRobot Runtime DLL 경로 추가
if hasattr(os, 'add_dll_directory'):
    dll_path = r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64"
    if os.path.exists(dll_path):
        os.add_dll_directory(dll_path)

# HIKRobot SDK
try:
    from MvImport.MvCameraControl_class import *
except ImportError:
    print("오류: HIKRobot SDK를 찾을 수 없습니다.")
    print("MvImport 폴더가 프로젝트에 있는지 확인하세요.")
    sys.exit(1)

# Dobot API
try:
    from dobot_api import DobotApiDashboard, DobotApiFeedBack
except ImportError:
    print("오류: dobot_api.py를 찾을 수 없습니다.")
    sys.exit(1)


class VLADataCollectorAuto:
    def __init__(self, robot_ip="192.168.5.1", calibration_file=None):
        """
        Args:
            robot_ip: 로봇 IP 주소
            calibration_file: 카메라 캘리브레이션 파일 (.npz)
        """
        self.robot_ip = robot_ip
        self.calibration_file = calibration_file
        
        # 카메라
        self.camera = None
        self.deviceList = None
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # 로봇
        self.dashboard = None
        self.feed = None
        
        # Teaching log 데이터
        self.log_data = []
        
        # VLA 데이터셋
        self.recorded_data = []
        self.frame_count = 0
        
        # 저장 경로
        self.session_name = f"vla_auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_dir = os.path.join("vla_dataset", self.session_name)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "images"), exist_ok=True)
        
        print(f"\n💾 데이터 저장 경로: {self.save_dir}")
    
    def load_calibration(self):
        """카메라 캘리브레이션 파일 로드"""
        if self.calibration_file and os.path.exists(self.calibration_file):
            try:
                data = np.load(self.calibration_file)
                self.camera_matrix = data['camera_matrix']
                self.dist_coeffs = data['dist_coeffs']
                print(f"✅ 캘리브레이션 로드: {self.calibration_file}")
                print(f"   RMS 오차: {data['rms_error']:.4f} pixels")
                return True
            except Exception as e:
                print(f"⚠️  캘리브레이션 로드 실패: {e}")
                print("   보정 없이 진행합니다.")
                return False
        else:
            print("⚠️  캘리브레이션 파일 없음. 보정 없이 진행합니다.")
            return False
    
    def init_camera(self):
        """HIKRobot 카메라 초기화"""
        print("\n" + "=" * 70)
        print("📷 카메라 초기화 중...")
        print("=" * 70)
        
        try:
            # SDK 초기화
            ret = MvCamera.MV_CC_Initialize()
            if ret != 0:
                print(f"오류: SDK 초기화 실패 (0x{ret:x})")
                return False
            
            # 카메라 검색
            self.deviceList = MV_CC_DEVICE_INFO_LIST()
            tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
            
            ret = MvCamera.MV_CC_EnumDevices(tlayerType, self.deviceList)
            if ret != 0:
                print(f"오류: 카메라 검색 실패 (0x{ret:x})")
                return False
            
            if self.deviceList.nDeviceNum == 0:
                print("오류: 연결된 카메라가 없습니다.")
                return False
            
            print(f"✅ 카메라 발견: {self.deviceList.nDeviceNum}개")
            
            # 카메라 열기
            self.camera = MvCamera()
            stDeviceInfo = cast(self.deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
            
            ret = self.camera.MV_CC_CreateHandle(stDeviceInfo)
            if ret != 0:
                print(f"오류: 카메라 핸들 생성 실패 (0x{ret:x})")
                return False
            
            ret = self.camera.MV_CC_OpenDevice()
            if ret != 0:
                print(f"오류: 카메라 열기 실패 (0x{ret:x})")
                return False
            
            # 카메라 설정
            self.camera.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
            
            # 자동 노출/게인
            try:
                self.camera.MV_CC_SetEnumValue("ExposureAuto", 2)  # Continuous
                self.camera.MV_CC_SetEnumValue("GainAuto", 2)  # Continuous
            except:
                pass
            
            # 스트리밍 시작
            ret = self.camera.MV_CC_StartGrabbing()
            if ret != 0:
                print(f"오류: 스트리밍 시작 실패 (0x{ret:x})")
                return False
            
            print("✅ 카메라 초기화 완료!")
            return True
            
        except Exception as e:
            print(f"오류: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_frame(self):
        """HIKRobot 카메라에서 프레임 가져오기"""
        try:
            buffer_size = 2448 * 2048 * 3
            pData = (c_ubyte * buffer_size)()
            stFrameInfo = MV_FRAME_OUT_INFO_EX()
            memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
            
            ret = self.camera.MV_CC_GetOneFrameTimeout(pData, buffer_size, stFrameInfo, 1000)
            
            if ret == 0:
                image_data = np.frombuffer(pData, dtype=np.uint8, count=stFrameInfo.nFrameLen)
                
                # Bayer 형식 변환
                if stFrameInfo.enPixelType == PixelType_Gvsp_BayerRG8:
                    image = image_data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
                    image = cv2.cvtColor(image, cv2.COLOR_BayerRG2BGR)
                elif stFrameInfo.enPixelType == PixelType_Gvsp_BayerGR8:
                    image = image_data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
                    image = cv2.cvtColor(image, cv2.COLOR_BayerGR2BGR)
                elif stFrameInfo.enPixelType == PixelType_Gvsp_BayerGB8:
                    image = image_data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
                    image = cv2.cvtColor(image, cv2.COLOR_BayerGB2BGR)
                elif stFrameInfo.enPixelType == PixelType_Gvsp_BayerBG8:
                    image = image_data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
                    image = cv2.cvtColor(image, cv2.COLOR_BayerBG2BGR)
                else:
                    if len(image_data) == stFrameInfo.nHeight * stFrameInfo.nWidth:
                        image = image_data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    else:
                        image = image_data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth, -1))
                
                # 640x480으로 리사이즈
                image = cv2.resize(image, (640, 480))
                
                # 캘리브레이션 보정 적용
                if self.camera_matrix is not None and self.dist_coeffs is not None:
                    image = cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
                
                # BGR ↔ RGB 변환 (색상 반전 수정)
                # HIKRobot 카메라는 RGB 순서인데 OpenCV는 BGR을 사용하므로 변환 필요
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                return True, image
            else:
                return False, None
                
        except Exception as e:
            return False, None
    
    def init_robot(self):
        """로봇 초기화"""
        print("\n" + "=" * 70)
        print("🤖 로봇 연결 중...")
        print("=" * 70)
        
        try:
            self.dashboard = DobotApiDashboard(self.robot_ip, 29999)
            self.feed = DobotApiFeedBack(self.robot_ip, 30004)
            
            time.sleep(0.5)
            print(f"✅ 로봇 연결 완료: {self.robot_ip}")
            
            return True
            
        except Exception as e:
            print(f"오류: 로봇 연결 실패 - {e}")
            return False
    
    def enable_robot(self):
        """로봇 활성화"""
        print("\n🔧 로봇 활성화 중...")
        
        try:
            # 에러 클리어
            self.dashboard.ClearError()
            time.sleep(0.5)
            
            # Enable
            result = self.dashboard.EnableRobot()
            print(f"   EnableRobot 결과: {result}")
            time.sleep(1)
            
            # 상태 확인
            for i in range(10):
                data = self.feed.feedBackData()
                if data is not None and len(data) > 0:
                    mode = data['RobotMode'][0]
                    if mode == 5:  # ROBOT_MODE_ENABLE
                        print(f"✅ 로봇 활성화 성공 (RobotMode: {mode})")
                        return True
                    elif mode == 7:  # ROBOT_MODE_RUNNING
                        print(f"✅ 로봇 준비 완료 (RobotMode: {mode})")
                        return True
                    print(f"   대기 중... (RobotMode: {mode})")
                time.sleep(0.5)
            
            print("⚠️ 로봇 활성화 시간 초과")
            return False
            
        except Exception as e:
            print(f"❌ 로봇 활성화 실패: {e}")
            return False
    
    def _wait_for_robot_stop(self, timeout=10.0):
        """로봇이 정지할 때까지 대기"""
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            try:
                data = self.feed.feedBackData()
                if data is not None and len(data) > 0:
                    # RunningStatus 확인: 0=정지, 1=실행중
                    running_status = data['RunningStatus'][0]
                    if running_status == 0:
                        # 추가 대기 (진동 감쇠)
                        time.sleep(0.1)
                        return True
            except:
                pass
            time.sleep(0.01)
        
        return False
    
    def list_log_files(self):
        """로그 폴더의 파일 목록 표시"""
        log_dir = "logs"
        
        if not os.path.exists(log_dir):
            print(f"⚠️ 로그 폴더가 없습니다: {log_dir}")
            return []
        
        files = [f for f in os.listdir(log_dir) if f.endswith('.csv')]
        
        if not files:
            print(f"⚠️ 로그 파일이 없습니다.")
            return []
        
        # 최신 파일 순으로 정렬
        files.sort(reverse=True)
        
        print("\n" + "=" * 70)
        print("📁 사용 가능한 티칭 로그:")
        print("=" * 70)
        
        for idx, filename in enumerate(files, 1):
            filepath = os.path.join(log_dir, filename)
            
            # CSV 읽어서 포인트 수 확인
            try:
                with open(filepath, 'r', encoding='utf-8-sig') as f:
                    line_count = sum(1 for line in f) - 1  # 헤더 제외
                print(f"  {idx}. {filename} ({line_count} 포인트)")
            except:
                print(f"  {idx}. {filename}")
        
        print("=" * 70)
        
        return files
    
    def load_log_file(self, filepath):
        """로그 파일 읽기"""
        print(f"\n📂 로그 파일 로딩: {filepath}")
        
        try:
            self.log_data = []
            
            with open(filepath, 'r', encoding='utf-8-sig') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row in reader:
                    entry = {
                        'j1': float(row['j1']),
                        'j2': float(row['j2']),
                        'j3': float(row['j3']),
                        'j4': float(row['j4']),
                        'j5': float(row['j5']),
                        'j6': float(row['j6']),
                        'x': float(row['x']),
                        'y': float(row['y']),
                        'z': float(row['z']),
                        'rx': float(row['rx']),
                        'ry': float(row['ry']),
                        'rz': float(row['rz']),
                        'robot_mode': int(row['robot_mode']),
                        'tooldo1': int(row.get('tooldo1', 0)),
                        'tooldo2': int(row.get('tooldo2', 0)),
                    }
                    
                    self.log_data.append(entry)
            
            print(f"✅ 로그 로딩 완료: {len(self.log_data)}개 포인트")
            return True
            
        except Exception as e:
            print(f"❌ 로그 로딩 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def replay_and_collect(self, mode='joint', speed_ratio=50, sample_skip=5):
        """
        로그 재생하면서 VLA 데이터 수집
        
        Args:
            mode: 'joint' (관절각) 또는 'cartesian' (직교좌표)
            speed_ratio: 속도 비율 (1-100)
            sample_skip: 샘플 건너뛰기 (1=모든 포인트, 5=5개 중 1개)
        """
        print("\n" + "=" * 70)
        print(f"🎬 VLA 데이터 수집 시작")
        print("=" * 70)
        print(f"⚙️  재생 모드: {'관절각 모드' if mode == 'joint' else '직교좌표 모드'}")
        print(f"⚙️  속도 비율: {speed_ratio}%")
        print(f"⚙️  샘플 간격: {sample_skip}")
        print("=" * 70)
        
        total_points = len(self.log_data)
        replay_indices = list(range(0, total_points, sample_skip))
        
        print(f"\n총 {len(replay_indices)}개 포인트를 재생합니다.")
        print("Ctrl+C를 눌러 중단할 수 있습니다.\n")
        
        try:
            for idx, data_idx in enumerate(replay_indices):
                point = self.log_data[data_idx]
                
                # 1. 로봇 이동
                if mode == 'joint':
                    # 관절각 모드
                    result = self.dashboard.MovJ(
                        point['j1'], point['j2'], point['j3'],
                        point['j4'], point['j5'], point['j6'],
                        1,  # coordinateMode=1 (관절각)
                        v=speed_ratio
                    )
                else:
                    # 직교좌표 모드
                    result = self.dashboard.MovJ(
                        point['x'], point['y'], point['z'],
                        point['rx'], point['ry'], point['rz'],
                        0,  # coordinateMode=0 (직교좌표)
                        v=speed_ratio
                    )
                
                # 2. 로봇 정지 대기 (중요!)
                self._wait_for_robot_stop(timeout=10.0)
                
                # 3. 그리퍼/석션 상태 재현
                try:
                    self.dashboard.ToolDOInstant(1, point['tooldo1'])
                    self.dashboard.ToolDOInstant(2, point['tooldo2'])
                except:
                    pass
                
                # 4. 정지 후 이미지 캡처 (블러 방지)
                time.sleep(0.05)  # 추가 안정화
                ret, frame = self.get_frame()
                
                if not ret or frame is None:
                    print(f"⚠️  [{idx+1}/{len(replay_indices)}] 프레임 캡처 실패")
                    continue
                
                # 5. 현재 로봇 피드백 읽기
                feed_data = self.feed.feedBackData()
                
                if feed_data is None or len(feed_data) == 0:
                    print(f"⚠️  [{idx+1}/{len(replay_indices)}] 피드백 읽기 실패")
                    continue
                
                # 6. 데이터 저장
                # 이미지 저장
                frame_filename = f"frame_{self.frame_count:06d}.jpg"
                frame_path = os.path.join(self.save_dir, "images", frame_filename)
                cv2.imwrite(frame_path, frame)
                
                # 첫 프레임 크기 확인
                if self.frame_count == 0:
                    print(f"💾 저장되는 이미지 크기: {frame.shape[1]}x{frame.shape[0]} (W x H)\n")
                
                # 데이터 기록
                joints = feed_data['QActual'][0].tolist()
                tcp_pose = feed_data['ToolVectorActual'][0].tolist()
                
                record = {
                    'frame_id': self.frame_count,
                    'timestamp': time.time(),
                    'image_path': frame_filename,
                    'joint_angles': joints,
                    'tcp_pose': tcp_pose,
                    'gripper_tooldo1': point['tooldo1'],
                    'gripper_tooldo2': point['tooldo2'],
                    'robot_mode': int(feed_data['RobotMode'][0])
                }
                
                self.recorded_data.append(record)
                self.frame_count += 1
                
                # 진행 상황 표시
                progress = (idx + 1) / len(replay_indices) * 100
                gripper_status = f"Gripper:[{point['tooldo1']},{point['tooldo2']}]"
                
                if mode == 'joint':
                    print(f"[{progress:5.1f}%] {idx+1}/{len(replay_indices)} - "
                          f"J1={point['j1']:7.2f}°, J2={point['j2']:7.2f}°, J3={point['j3']:7.2f}° {gripper_status}")
                else:
                    print(f"[{progress:5.1f}%] {idx+1}/{len(replay_indices)} - "
                          f"X={point['x']:7.2f}, Y={point['y']:7.2f}, Z={point['z']:7.2f} {gripper_status}")
                
                # 에러 체크
                if "Not Tcp" in str(result):
                    print(f"⚠️  경고: TCP 모드가 아닙니다. - {result}")
                    break
            
            print("\n" + "=" * 70)
            print("✅ 데이터 수집 완료!")
            print("=" * 70)
            
        except KeyboardInterrupt:
            print("\n\n⚠️ 사용자가 중단했습니다.")
        except Exception as e:
            print(f"\n❌ 수집 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
    
    def save_data(self):
        """수집된 데이터 저장"""
        print("\n" + "=" * 70)
        print("💾 VLA 데이터셋 저장 중...")
        print("=" * 70)
        
        if len(self.recorded_data) == 0:
            print("⚠️  수집된 데이터가 없습니다.")
            return
        
        # CSV 파일로 저장
        csv_path = os.path.join(self.save_dir, "robot_data.csv")
        with open(csv_path, 'w') as f:
            # 헤더
            f.write("frame_id,timestamp,image_path,")
            f.write("j1,j2,j3,j4,j5,j6,")
            f.write("x,y,z,rx,ry,rz,")
            f.write("gripper_tooldo1,gripper_tooldo2,robot_mode\n")
            
            # 데이터
            for record in self.recorded_data:
                f.write(f"{record['frame_id']},{record['timestamp']},{record['image_path']},")
                f.write(','.join(map(str, record['joint_angles'])) + ',')
                f.write(','.join(map(str, record['tcp_pose'])) + ',')
                f.write(f"{record['gripper_tooldo1']},{record['gripper_tooldo2']},{record['robot_mode']}\n")
        
        print(f"✅ CSV 저장: {csv_path}")
        
        # NumPy 포맷으로 저장
        npy_path = os.path.join(self.save_dir, "dataset.npy")
        np.save(npy_path, self.recorded_data)
        print(f"✅ NumPy 저장: {npy_path}")
        
        # 메타데이터 저장
        meta_path = os.path.join(self.save_dir, "metadata.txt")
        
        # 실제 저장된 첫 이미지 크기 확인
        actual_resolution = "640x480"
        if len(self.recorded_data) > 0:
            first_img_path = os.path.join(self.save_dir, "images", self.recorded_data[0]['image_path'])
            if os.path.exists(first_img_path):
                test_img = cv2.imread(first_img_path)
                if test_img is not None:
                    actual_resolution = f"{test_img.shape[1]}x{test_img.shape[0]}"
        
        with open(meta_path, 'w') as f:
            f.write(f"VLA Dataset - Auto Collection\n")
            f.write(f"=" * 70 + "\n\n")
            f.write(f"Session: {self.session_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Frames: {len(self.recorded_data)}\n")
            f.write(f"Robot IP: {self.robot_ip}\n")
            f.write(f"Calibration: {self.calibration_file if self.calibration_file else 'None'}\n")
            f.write(f"Image Resolution: {actual_resolution}\n")
            f.write(f"\nData Collection Method: Teaching Log Replay\n")
            f.write(f"- Teaching log를 자동 재생하면서 촬영\n")
            f.write(f"- 각 포인트에서 로봇 정지 후 이미지 캡처\n")
            f.write(f"- 정확한 타임스텝 동기화 보장\n")
        
        print(f"✅ 메타데이터 저장: {meta_path}")
        
        print("\n" + "=" * 70)
        print("✅ 모든 데이터 저장 완료!")
        print("=" * 70)
        print(f"\n저장 위치: {self.save_dir}")
        print(f"수집된 프레임: {len(self.recorded_data)}장")
        print(f"\n파일 목록:")
        print(f"  - robot_data.csv: 로봇 데이터 (CSV)")
        print(f"  - dataset.npy: 통합 데이터 (NumPy)")
        print(f"  - images/: 프레임 이미지 ({len(self.recorded_data)}장)")
        print(f"  - metadata.txt: 메타데이터")
    
    def cleanup(self):
        """리소스 정리"""
        print("\n리소스 정리 중...")
        
        # 카메라 종료
        if self.camera:
            try:
                self.camera.MV_CC_StopGrabbing()
                self.camera.MV_CC_CloseDevice()
                self.camera.MV_CC_DestroyHandle()
            except:
                pass
        
        try:
            MvCamera.MV_CC_Finalize()
        except:
            pass
        
        # 로봇 연결 종료
        if self.dashboard:
            try:
                self.dashboard.close()
            except:
                pass
        
        if self.feed:
            try:
                self.feed.close()
            except:
                pass
        
        print("✅ 정리 완료")


def main():
    print("\n" + "=" * 70)
    print("🤖 VLA 데이터 자동 수집 프로그램")
    print("=" * 70)
    print("\n이 프로그램은 티칭 로그를 자동 재생하면서 VLA 데이터를 수집합니다.")
    print("\n워크플로우:")
    print("  1. logs/ 폴더에서 티칭 로그 선택")
    print("  2. 물체를 초기 위치에 세팅")
    print("  3. 사람은 카메라 시야 밖으로 이동")
    print("  4. 로봇이 자동으로 재생하면서 각 포인트에서 촬영")
    print("  5. VLA 데이터셋 생성 (이미지 + 로봇 상태)")
    print("=" * 70)
    
    # 로봇 IP 주소
    robot_ip = "192.168.5.1"
    
    # 캘리브레이션 파일
    calibration_file = None
    if os.path.exists("hikrobot_calibration_20260126_143821.npz"):
        calibration_file = "hikrobot_calibration_20260126_143821.npz"
    
    # 데이터 수집기 생성
    collector = VLADataCollectorAuto(robot_ip=robot_ip, calibration_file=calibration_file)
    
    try:
        # 1. 티칭 로그 선택
        files = collector.list_log_files()
        
        if not files:
            print("\n❌ 사용 가능한 로그 파일이 없습니다.")
            print("   먼저 manual_teaching_logger.py로 로그를 만드세요.")
            return
        
        while True:
            try:
                choice = input("\n📝 재생할 파일 번호를 입력하세요 (0=취소): ")
                
                if choice == '0':
                    print("취소되었습니다.")
                    return
                
                idx = int(choice) - 1
                
                if 0 <= idx < len(files):
                    filepath = os.path.join("logs", files[idx])
                    break
                else:
                    print("⚠️ 잘못된 번호입니다.")
            except ValueError:
                print("⚠️ 숫자를 입력하세요.")
        
        # 2. 로그 파일 로딩
        if not collector.load_log_file(filepath):
            return
        
        # 3. 재생 모드 선택
        print("\n" + "=" * 70)
        print("🎯 재생 모드 선택:")
        print("=" * 70)
        print("  1. 관절각 모드 - 원래 관절 각도를 재현")
        print("  2. 직교좌표 모드 - 원래 좌표 위치를 재현")
        print("=" * 70)
        
        while True:
            mode_choice = input("\n재생 모드를 선택하세요 (1 또는 2, Enter=1): ")
            
            if mode_choice == '' or mode_choice == '1':
                mode = 'joint'
                break
            elif mode_choice == '2':
                mode = 'cartesian'
                break
            else:
                print("⚠️ 1 또는 2를 입력하세요.")
        
        # 4. 속도 설정
        print("\n" + "=" * 70)
        print("⚡ 속도 설정:")
        print("=" * 70)
        print("  - 권장: 30-50% (안전)")
        print("=" * 70)
        
        while True:
            try:
                speed_input = input("\n속도 비율을 입력하세요 (1-100%, Enter=50%): ")
                
                if speed_input == '':
                    speed_ratio = 50
                    break
                
                speed_ratio = int(speed_input)
                
                if 1 <= speed_ratio <= 100:
                    break
                else:
                    print("⚠️ 1-100 사이의 값을 입력하세요.")
            except ValueError:
                print("⚠️ 숫자를 입력하세요.")
        
        # 5. 샘플링 간격 설정
        print("\n" + "=" * 70)
        print("📊 샘플링 간격 설정:")
        print("=" * 70)
        print("  - 1: 모든 포인트 (가장 많은 데이터, 느림)")
        print("  - 5: 5개 중 1개 (추천, 균형)")
        print("  - 10: 10개 중 1개 (빠름, 적은 데이터)")
        print("=" * 70)
        
        while True:
            try:
                skip_input = input("\n샘플 간격을 입력하세요 (1-20, Enter=5): ")
                
                if skip_input == '':
                    sample_skip = 5
                    break
                
                sample_skip = int(skip_input)
                
                if 1 <= sample_skip <= 20:
                    break
                else:
                    print("⚠️ 1-20 사이의 값을 입력하세요.")
            except ValueError:
                print("⚠️ 숫자를 입력하세요.")
        
        # 6. 최종 확인
        print("\n" + "=" * 70)
        print("📋 설정 확인:")
        print("=" * 70)
        print(f"  파일: {files[idx]}")
        print(f"  모드: {'관절각 모드' if mode == 'joint' else '직교좌표 모드'}")
        print(f"  속도: {speed_ratio}%")
        print(f"  샘플 간격: {sample_skip}")
        print(f"  예상 프레임 수: 약 {len(collector.log_data) // sample_skip}장")
        print("=" * 70)
        
        print("\n⚠️  중요:")
        print("  1. 물체를 초기 위치에 세팅하세요")
        print("  2. 카메라 시야 밖으로 이동하세요")
        print("  3. 로봇 주변에 장애물이 없는지 확인하세요")
        
        input("\n준비되었으면 Enter 키를 누르세요...")
        
        # 7. 캘리브레이션 로드
        collector.load_calibration()
        
        # 8. 카메라 초기화
        if not collector.init_camera():
            print("\n❌ 카메라 초기화 실패!")
            return
        
        # 9. 로봇 초기화
        if not collector.init_robot():
            print("\n❌ 로봇 초기화 실패!")
            return
        
        # 10. 로봇 활성화
        if not collector.enable_robot():
            print("\n❌ 로봇 활성화 실패!")
            return
        
        # 11. 데이터 수집 시작
        collector.replay_and_collect(
            mode=mode,
            speed_ratio=speed_ratio,
            sample_skip=sample_skip
        )
        
        # 12. 데이터 저장
        collector.save_data()
        
    except KeyboardInterrupt:
        print("\n\n사용자가 중단했습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
    finally:
        collector.cleanup()


if __name__ == "__main__":
    main()
