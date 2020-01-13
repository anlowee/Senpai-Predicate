# Senpai-Predicate
↑↑↓↑↓↓↑↓↑
# Senpai-Predicate
↑↑↓↑↓↓↑↓↑

# sample code
```(python)
import time
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# load the moudle
start = time.clock()
# you can fix the model name here
model = load_model('model.h5')
print('Warming up took {}s'.format(time.clock() - start))

# preprocessing
# you can replace the path with the image to predicate
path = 'some image path'
img_height, img_width = 224, 224
x = image.load_img(path=path, target_size=(img_height, img_width))
x = image.img_to_array(x)
x = x[None]

# predicate
start = time.clock()
y = model.predict(x)
print('Prediction took {}s'.format(time.clock() - start))

# confidence
for i in np.argsort(y[0])[::-1][:5]:
    print('{}:{:.2f}%'.format(i, y[0][i] * 100))
```
