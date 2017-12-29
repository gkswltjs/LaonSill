 본 문서는 Yolo v2의 pretrain weight를 얻고, 그것을 LaonSill의 param으로 저장하는 방법에 
 대해서 기술합니다.

 * Yolov2 pretrain weight를 LaonSill param으로 변경하는 방법
  (1) npz 파일로 된 pretrain weight를 다운받는다.
    - 본인의 경우 pytorch 소스로 구현된 https://github.com/longcw/yolo2-pytorch 소스에서
      다운 받았음.
    - 다운로드 경로 :
      https://drive.google.com/file/d/0B4pXCfnYmG1WRG52enNpcV80aDg/view?usp=sharing
    - 회사 NAS 서버에도 올려 두었음.
      /Result/network/legacy/YOLOv2-ILSVRC-PRETRAIN_PYTORCH/darknet19.weights.npz

  (2) convert_npz2laonsill_param.py 파일을 열어서 다운받은 npz파일의 경로를 NPZ_FILEPATH에
     기입한다. 저장이 될 laonsill param의 경로는 PARAM_FILEPATH에 저장한다.

  (3) convert_npz2laonsill_param.py를 실행한다.
      (※ 참고로 결과물은 NAS 서버의
       /Result/network/legacy/YOLOv2-ILSVRC-PRETRAIN_SOOOA/darknet_pretrain.param 으로 올려 
       두었습니다.)
