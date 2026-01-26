"""
VLA (Vision-Language-Action) 모델 학습용 데이터 수집 프로그램

HIKRobot 카메라 + Dobot 로봇 통합
- 카메라 이미지 (640x480, 캘리브레이션 보정 적용)
- 로봇 데이터 (관절각, 위치, 그리퍼 상태)
- 타임스탬프 동기화

사용 방법:
1. 로봇을 백드라이브 모드로 전환
2. python vla_data_collector.py
3. 물체 세팅 → 카메라 시야 밖으로 이동
4. 's' 키: 기록 시작/재개
5. 로봇을 움직여서 작업 수행
6. 'p' 키: 기록 일시정지
7. 반복...
8. 'q' 키: 종료 및 저장
"""

import sys
import os
import cv2
import numpy as np
from ctypes import *
import time
import threading
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
    from dobot_api import DobotApiDashboard, DobotApiFeedBack, DobotApi
except ImportError:
    print("오류: dobot_api.py를 찾을 수 없습니다.")
    sys.exit(1)


class VLADataCollector:
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
        self.feed_thread = None
        self.feed_running = False
        
        # 데이터 수집
        self.is_recording = False
        self.recorded_data = []
        self.frame_count = 0
        
        # 동기화
        self.lock = threading.Lock()
        self.latest_robot_data = None
        
        # 저장 경로
        self.session_name = f"vla_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_dir = os.path.join("vla_dataset", self.session_name)
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "images"), exist_ok=True)
        
        print(f"\n데이터 저장 경로: {self.save_dir}")
    
    def load_calibration(self):
        """카메라 캘리브레이션 파일 로드"""
        if self.calibration_file and os.path.exists(self.calibration_file):
            try:
                data = np.load(self.calibration_file)
                self.camera_matrix = data['camera_matrix']
                self.dist_coeffs = data['dist_coeffs']
                print(f"✅ 캘리브레이션 파일 로드: {self.calibration_file}")
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
            
            # 프레임레이트
            self.camera.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
            self.camera.MV_CC_SetFloatValue("AcquisitionFrameRate", 10.0)
            
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
            
            # 에러 클리어
            self.dashboard.ClearError()
            time.sleep(0.5)
            
            print(f"✅ 로봇 연결 완료: {self.robot_ip}")
            
            # 피드백 스레드 시작
            self.feed_running = True
            self.feed_thread = threading.Thread(target=self._feed_loop, daemon=True)
            self.feed_thread.start()
            
            return True
            
        except Exception as e:
            print(f"오류: 로봇 연결 실패 - {e}")
            return False
    
    def _feed_loop(self):
        """로봇 피드백 수집 루프"""
        while self.feed_running:
            try:
                data = self.feed.feedBackData()
                if data is not None and len(data) > 0:
                    with self.lock:
                        self.latest_robot_data = {
                            'timestamp': time.time(),
                            'joint_angles': data['QActual'][0].tolist() if 'QActual' in data.dtype.names else [0]*6,
                            'tcp_pose': data['ToolVectorActual'][0].tolist() if 'ToolVectorActual' in data.dtype.names else [0]*6,
                            'robot_mode': int(data['RobotMode'][0]) if 'RobotMode' in data.dtype.names else 0,
                        }
                        
                        # ToolDO 상태 (그리퍼)
                        try:
                            tool_do_result = self.dashboard.GetToolDO(1)
                            if tool_do_result and len(tool_do_result) > 0:
                                parts = tool_do_result.split(',')
                                if len(parts) >= 3:
                                    self.latest_robot_data['gripper'] = int(parts[2])
                                else:
                                    self.latest_robot_data['gripper'] = 0
                            else:
                                self.latest_robot_data['gripper'] = 0
                        except:
                            self.latest_robot_data['gripper'] = 0
                
                time.sleep(0.01)  # 100Hz
                
            except Exception as e:
                time.sleep(0.1)
    
    def collect_data(self):
        """메인 데이터 수집 루프"""
        print("\n" + "=" * 70)
        print("🎥 VLA 데이터 수집 시작")
        print("=" * 70)
        print("\n조작 방법:")
        print("  's' 키 - 기록 시작/재개 (Start/Resume)")
        print("  'p' 키 - 기록 일시정지 (Pause)")
        print("  'q' 키 - 종료 및 저장 (Quit)")
        print("\n워크플로우:")
        print("  1. 물체를 세팅하고 카메라 시야 밖으로 이동")
        print("  2. 's' 키를 눌러 기록 시작")
        print("  3. 로봇을 백드라이브로 움직여서 작업 수행")
        print("  4. 작업 완료 후 'p' 키를 눌러 일시정지")
        print("  5. 물체 재배치 후 반복...")
        print("=" * 70)
        print("\n⏸️  일시정지 상태 (물체를 세팅하세요)")
        
        while True:
            # 카메라 프레임 가져오기
            ret, frame = self.get_frame()
            
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            
            # 화면 표시용 프레임
            display_frame = frame.copy()
            
            # 기록 상태 표시
            if self.is_recording:
                # 빨간 점 (녹화 중)
                cv2.circle(display_frame, (30, 30), 15, (0, 0, 255), -1)
                cv2.putText(display_frame, "RECORDING", (55, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # 데이터 수집
                with self.lock:
                    if self.latest_robot_data is not None:
                        # 프레임 저장 (크기 확인)
                        frame_filename = f"frame_{self.frame_count:06d}.jpg"
                        frame_path = os.path.join(self.save_dir, "images", frame_filename)
                        
                        # 이미지 크기 확인 (첫 번째 프레임만)
                        if self.frame_count == 0:
                            print(f"\n💾 저장되는 이미지 크기: {frame.shape[1]}x{frame.shape[0]} (W x H)")
                        
                        cv2.imwrite(frame_path, frame)
                        
                        # 데이터 기록
                        record = {
                            'frame_id': self.frame_count,
                            'timestamp': self.latest_robot_data['timestamp'],
                            'image_path': frame_filename,
                            'joint_angles': self.latest_robot_data['joint_angles'],
                            'tcp_pose': self.latest_robot_data['tcp_pose'],
                            'gripper': self.latest_robot_data['gripper'],
                            'robot_mode': self.latest_robot_data['robot_mode']
                        }
                        
                        self.recorded_data.append(record)
                        self.frame_count += 1
            else:
                # 회색 점 (일시정지)
                cv2.circle(display_frame, (30, 30), 15, (128, 128, 128), -1)
                cv2.putText(display_frame, "PAUSED", (55, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
            
            # 프레임 카운트 표시
            cv2.putText(display_frame, f"Frames: {self.frame_count}", (10, 470),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 로봇 상태 표시
            with self.lock:
                if self.latest_robot_data is not None:
                    tcp = self.latest_robot_data['tcp_pose']
                    gripper = self.latest_robot_data['gripper']
                    cv2.putText(display_frame, f"TCP: X={tcp[0]:.1f} Y={tcp[1]:.1f} Z={tcp[2]:.1f}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    cv2.putText(display_frame, f"Gripper: {'ON' if gripper else 'OFF'}", 
                               (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow('VLA Data Collection', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # 's' 키: 기록 시작/재개
            if key == ord('s'):
                if not self.is_recording:
                    self.is_recording = True
                    print("\n🔴 기록 시작!")
            
            # 'p' 키: 기록 일시정지
            elif key == ord('p'):
                if self.is_recording:
                    self.is_recording = False
                    print(f"\n⏸️  일시정지 (현재 {self.frame_count}장 수집)")
            
            # 'q' 키: 종료
            elif key == ord('q'):
                print("\n종료 중...")
                break
        
        cv2.destroyAllWindows()
    
    def save_data(self):
        """수집된 데이터 저장"""
        print("\n" + "=" * 70)
        print("💾 데이터 저장 중...")
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
            f.write("gripper,robot_mode\n")
            
            # 데이터
            for record in self.recorded_data:
                f.write(f"{record['frame_id']},{record['timestamp']},{record['image_path']},")
                f.write(','.join(map(str, record['joint_angles'])) + ',')
                f.write(','.join(map(str, record['tcp_pose'])) + ',')
                f.write(f"{record['gripper']},{record['robot_mode']}\n")
        
        print(f"✅ CSV 저장: {csv_path}")
        
        # NumPy 포맷으로 저장 (.npy)
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
            f.write(f"VLA Dataset Collection\n")
            f.write(f"=" * 70 + "\n\n")
            f.write(f"Session: {self.session_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Frames: {len(self.recorded_data)}\n")
            f.write(f"Robot IP: {self.robot_ip}\n")
            f.write(f"Calibration: {self.calibration_file if self.calibration_file else 'None'}\n")
            f.write(f"Image Resolution: {actual_resolution} (실제 저장됨)\n")
        
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
        
        # 피드백 스레드 종료
        self.feed_running = False
        if self.feed_thread:
            self.feed_thread.join(timeout=1.0)
        
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
        
        cv2.destroyAllWindows()
        print("✅ 정리 완료")


def main():
    print("\n" + "=" * 70)
    print("🤖 VLA 데이터 수집 프로그램")
    print("=" * 70)
    
    # 로봇 IP 주소
    robot_ip = "192.168.5.1"
    
    # 최신 캘리브레이션 파일 자동 찾기
    calibration_file = None
    if os.path.exists("hikrobot_calibration_20260126_143821.npz"):
        calibration_file = "hikrobot_calibration_20260126_143821.npz"
    
    # 데이터 수집기 생성
    collector = VLADataCollector(robot_ip=robot_ip, calibration_file=calibration_file)
    
    try:
        # 캘리브레이션 로드
        collector.load_calibration()
        
        # 카메라 초기화
        if not collector.init_camera():
            print("\n카메라 초기화 실패!")
            return
        
        # 로봇 초기화
        if not collector.init_robot():
            print("\n로봇 초기화 실패!")
            return
        
        # 로봇 모드 확인
        print("\n⚠️  로봇을 백드라이브 모드로 전환하세요!")
        print("   (티칭 펜던트 또는 DobotStudio Pro에서 설정)")
        print("\n5초 후 자동으로 시작합니다...")
        time.sleep(5)
        
        # 데이터 수집 시작
        collector.collect_data()
        
        # 데이터 저장
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
