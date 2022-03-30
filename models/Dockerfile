FROM tensorflow/serving

ENV MODEL_BASE_PATH /model
ENV MODEL_NAME tfserving_classifier

COPY model /model
 
COPY tf_serving_entrypoin.sh user/bin/tf_serving_entrypoin.sh 
RUN chmod +x user/bin/tf_serving_entrypoin.sh 
ENTRYPOINT []
CMD ['user/bin/tf_serving_entrypoin.sh ']