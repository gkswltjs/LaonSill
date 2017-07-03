* 필수 준비 사항
(1) env.sh.eg에 명시된 환경 변수들이 제대로 설정이 되어 있어야 한다.
(2) $SOOOA_BUILD_PATH가 지정이 되어 있어야 한다.
 - export SOOOA_BUILD_PATH=/home/monhoney/git/neuralnetworksanddeeplearning/NeuralNetworksAndDeepLearning
(3) $SOOOA_BUILD_PATH에 build directory가 지워져 있어야 한다.
 - $SOOOA_BUILD_PATH/DebugGen, $SOOOA_BUILD_PATH/ReleaseGen, $SOOOA_BUILD_PATH/DebugClientGen,
   $SOOOA_BUILD_PATH/ReleaseClientGen 

* build.sh 사용 방법
 - build.sh dop [Debug|Release]
 - dop는 몇개의 멀티쓰레드로 빌드할 것인지를 명시 한다.
 - Debug 혹은 Release로 빌드시에 해당 문자열을 기입 한다. 
 - Debug|Release를 선택하지 않은 경우에 Debug, Releae 모드 빌드 한다.

* genMakefile.py 설정 방법
 (1) symbol 추가
   - symbolDic을 수정
 (2) lib 추가
   - libList에 추가
 (3) binary 이름 변경
   - targetNameDic 변경
   - build.sh도 바뀐 바이너리에 맞게 변경이 필요
 (4) build directory명 변경
   - dirNameDic을 변경
   - build.sh도 바뀐 바이너리에 맞게 변경이 필요
 (5) 지원하는 CUDA GPU device의 Arch, Code 변경
   - supportArch, supportCode 변경

