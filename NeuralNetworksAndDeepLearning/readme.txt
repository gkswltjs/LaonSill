본 문서는 본 프로젝트를 빌드하기 위해서 필요한 설정을 담고 있다.

* 초기 설정
(1) CUDA 경로
 - /usr/local/cuda
   (CUDA 설치시에 옵션으로 link를 걸도록 묻는데 이때 yes를 해주면 된다.)

(2) GNUPLOT_STREAM include 경로
 - INC_PATH_GNUPLOT 환경변수에 저장

(3) CIMG include 경로
 - INC_PATH_CIMG 환경변수에 저장

(4) SOOOA_HOME, SOOOA_SOURCE_PATH, SOOOA_BUILD_PATH 경로 설정

※ 이 외에도 추가할 설정이 있을 수 있다. 그때 그때 추가하고 문서도 갱신하도록
하자.

* 예제
(1) env.sh.eg를 env.sh로 복사한다.
 $ cp env.sh.eg env.sh

(2) env.sh를 자신의 환경에 맞게 수정한다. 
 [env.sh]
 export INC_PATH_GNUPLOT="/home/monhoney/git/gnuplot_iostream/gnuplot-iostream"
 export INC_PATH_CIMG="/home/monhoney/git/cImg/CImg"
 export SOOOA_HOME="/home/monhoney/SOOOA_HOME"
 export SOOOA_SOURCE_PATH="/home/monhoney/git/neuralnetworksanddeeplearning/NeuralNetworksAndDeepLearning/src"
 export SOOOA_BUILD_PATH="/home/monhoney/git/neuralnetworksanddeeplearning/NeuralNetworksAndDeepLearning"

(3) env.sh의 환경변수를 적용한다.
 $ source env.sh

(4) nsignt를 실행한다. 
 $ nsight

※ 최초의 설정의 경우에만 (1)~(2)의 과정을 수행하면 된다. 이후에는 (3),(4)
 과정으로 진행하면 된다. 물론 (3)의 과정을 .bashrc에 넣어서 자동으로 실행이
 되는 방법을 추천한다.
