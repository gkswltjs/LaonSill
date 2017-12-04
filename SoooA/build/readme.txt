* 필수 준비 사항
(1) env.sh.eg에 명시된 환경 변수들이 제대로 설정이 되어 있어야 한다.
(2) $LAONSILL_BUILD_PATH가 지정이 되어 있어야 한다.
$ export LAONSILL_BUILD_PATH=/home/monhoney/git/laonsill/LaonSill
(3) $LAONSILL_BUILD_PATH에 build directory가 지워져 있어야 한다.
$ ./cleanBuildGen.sh

* build.sh 사용 방법
 - build.sh dop [debug|release|tool|lib]
 - dop는 몇개의 멀티쓰레드로 빌드할 것인지를 명시 한다.
 - debug, release, tool, 혹은 lib 으로 빌드시에 해당 문자열을 기입 한다. 
 - 선택하지 않은 경우에 전체 모드 빌드 한다.

