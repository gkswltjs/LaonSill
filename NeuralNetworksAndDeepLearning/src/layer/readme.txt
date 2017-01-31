※ 주의사항
메소드는 편의상 반환값과 인자들을 생략하고 "메쏘드이름()" 형태로 기입하였다.
변수와 클래스 역시 변수명과 클래스명으로 기입하였다.

* Layer 추가시에 필요한 작업
본 문서는 AAA 라는 신규 Layer 추가시에 필요한 동작을 서술하였다.

(1) Layer.h에 AAA LayerType 등록
 [Layer.h]
	enum LayerType : int {
		None = 0,
		Input, 					// 입력 레이어
        ALEInput,               // ALE 입력 레이어
		FullyConnected, 		// Fully Connected 레이어
              :
              :
        AAA
    };

(2) 3개의 파일을 만든다.
  - AAALayer.h : AAA Layer header 파일
  - AAALayer.cpp : AAA Layer source 파일. CPU + 공통 코드들
  - AAALayer_device.cu : AAA Layer의 GPU 관련 코드들
  
(3) AAA Layer의 클래스를 선언한다. 이때에 상속해야할 클래스는 아래와 같다:
  - 가장 앞단의 Layer인 경우 => InputLayer를 상속
  - 가장 앞단의 Layer가 아닌 다른 Layer인 경우 => HiddenLayer를 상속
  - 가장 뒷단의 Layer인 경우 => OutputLayer를 상속
  - 학습이 필요한 Layer(ex. FullyConnecetdLayer, ConvLayer) => LearnableLayer를 상속
  - 그리고, 모든 Layer들은 Layer class를 상속받는다.

(4) builder class를 Layer의 inner class로 생성한다.
  - AAA Layer를 생성하기 위한 클래스이다.
  - AAA Layer class 생성에 필요한 변수들을 선언하고, 그것을 변경할 수 있는 setter method를 
    제공한다. (ex. _var1, _var2, var1(), var2())
  - 생성자에서 default값을 설정한다.
  - build()라는 메소드를 만들고, build() 메소드는 AAA Layer를 생성하여 반환하도록 한다.
  - name, id, inputs, outputs, propDown 메소드들도 적당히 부모 메소드를 호출하도록 맞춰준다.

[AAALayer.h]
template <typename Dtype>
class AAALayer : public HiddenLayer<Dtype>, public LearnableLayer<Dtype> {
public: 

	class Builder : public HiddenLayer<Dtype>::Builder {
	public:
		typename Activation<Dtype>::Type _activationType;

		uint32_t _var1;
        double   _var2;
                                        
		Builder() {
			this->type = Layer<Dtype>::AAA;
            _var1 = 0;
            _var2 = 0.0;
		}
		virtual Builder* name(const std::string name) {
			HiddenLayer<Dtype>::Builder::name(name);
			return this;
		}
		virtual Builder* id(uint32_t id) {
			HiddenLayer<Dtype>::Builder::id(id);
			return this;
		}
		virtual Builder* inputs(const std::vector<std::string>& inputs) {
			HiddenLayer<Dtype>::Builder::inputs(inputs);
			return this;
		}
		virtual Builder* outputs(const std::vector<std::string>& outputs) {
			HiddenLayer<Dtype>::Builder::outputs(outputs);
			return this;
		}
		virtual Builder* propDown(const std::vector<bool>& propDown) {
			HiddenLayer<Dtype>::Builder::propDown(propDown);
			return this;
		}
		virtual Builder* var1(uint32_t var1) {
			this->_var1 = var1;
			return this;
		}
		virtual Builder* var2(double var2) {
			this->_var2 = var2;
			return this;
		}
		Layer<Dtype>* build() {
			return new AAALayer(this);
		}
	};

      :
      :

(5) AAA Layer에서 상속받은 클래스에 대한 메소드들을 구현한다. 상속받은 클래스에 대한 구현이
   필요한 메소드들은 아래와 같다:
   (모든 메소드를 나열하지 않았다. 코드는 계속 변하기 때문에 직접 소스코드를 보고 확인해야
    한다.)
  - Learnable Layer
    => getName(), update(), sumSquareParamsData(), sumSquareParamsGrad(), scaleParamsGrad(),
       boundParams(), numParams(), saveParams(), loadParams()           ------> 필수O
    => _updateParam(), syncMutableMem(), applyChanges(), syncParams()   ------> 필수X
  - Hidden Layer
    => backpropagation(), reshape()                                     ------> 필수X
  - Layer
    => reshape(), feedforward(), ...                                    ------> 필수X
  - InputLayer => Layer 클래스의 메소드 구현 필요
  - OutputLayer => Learnable Layer, Hidden Layer, Layer 클래스의 메소드 구현 필요
    (※  필수O는 추상 method라서 반드시 상속받은 Layer에서 구현이 필요한 것을 의미한다.
       필수X는 반드시 필요하지는 않지만 아마도 구현을 해야할 메소드들을 열거하였다.)

(6) AAA Layer의 생성자(constructor)와 파괴자(deconstructor)를 만든다.
  - 본 프로젝트에서 생성자는 .cpp파일에만 만들고, initialize() 메소드를 만들어서 초기화 한다.
  - 본 프로젝트에서 파괴자는 .cu파일과 .cpp파일에 나누어 만든다. GPU관련 파괴자는 .cu파일,
   CPU관련 파괴자는 .cpp 파일에 구현한다.
    (※  관례적으로 위와 같이 수행을 한 것이다. 반드시 따를 필요는 없다.)

(7) (1)~(6)까지 과정을 완료한 이후, AAA Layer에서 필요한 동작들을 구현하면 된다.

마지막으로, AAALayer.h, AAALayer.cpp, AAALayer_device.cu 파일을 template 용도로 만들었으니
신규 레이어를 생성할 때에 참고하길 바란다.
