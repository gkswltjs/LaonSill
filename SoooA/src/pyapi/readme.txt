본 문서는 SoooA python 버전의 사용방법에 대해서 기술한다.

* SoooA Client Library 사용 방법

(1) SoooA library를 빌드한다.
$ cd build
$ ./cleanBuildGen.sh
$ ./build_only.sh 12 lib

(2) SoooA Client Library를 등록한다.
 - 예제) clinet library가 /home/monhoney/soooa/SoooA/dev/client/lib에 있는 경우
$ sudo vi /etc/ld.so.conf
/home/monhoney/soooa/SoooA/dev/client/lib

(3) $PYTHONPATH 환경변수에 SoooA pyapi 모듈을 추가한다.
$ export PYTHONPATH=$SOOOA_SOURCE_PATH/pyapi:$PYTHONPATH


