FROM tensorflow/serving

ENV MODEL_BASE_PATH /models
ENV MODEL_NAME tfserving_classifier

COPY models /models
 
COPY tf_serving_entrypoint.sh user/bin/tf_serving_entrypoin.sh
RUN chmod +x user/bin/tf_serving_entrypoin.sh 
ENTRYPOINT []
CMD ['user/bin/tf_serving_entrypoin.sh']