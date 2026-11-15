# 최적화
- 최적화: 목적 함수의 결괏값을 최적화하는 변수를 찾는 알고리즘

## 넘파이를 사용한 선형회귀
$$\hat{y} = Wx+b\\
MSE(W, b) = \frac{1}{n}\sum_{i=1}^n (y_i-\hat{y_i})^2\\
W_{i+1} = W_i - \alpha \frac{\partial}{\partial W} MSE(W, b)\\
b_{i+1} = b_i - \alpha \frac{\partial}{\partial W}$$
```py
import numpy as np

x = np.array(
    [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
    [11], [12], [13], [14], [15], [16], [17], [18], [19], [20],
    [21], [22], [23], [24], [25], [26], [27], [28], [29], [30]]
)
y = np.array(
    [[0.94], [1.98], [2.88], [3.92], [3.96], [4.55], [5.64], [6.3], [7.44], [9.1],
    [8.46], [9.5], [10.67], [11.16], [14], [11.83], [14.4], [14.25], [16.2], [16.32],
    [17.46], [19.8], [18], [21.34], [22], [22.5], [24.57], [26.04], [21.6], [28.8]]
)


weight = 0.0
bias = 0.0
learning_rate = 0.005


for epoch in range(10000):
    y_hat = weight * x + bias

    cost = ((y - y_hat) ** 2).mean()

    weight = weight - learning_rate * ((y_hat - y) * x).mean()
    bias = bias - learning_rate * (y_hat - y).mean()

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch : {epoch+1:4d}, Weight : {weight:.3f}, Bias : {bias:.3f}, Cost : {cost:.3f}")
```
```
Epoch : 1000, Weight : 0.872, Bias : -0.290, Cost : 1.377
Epoch : 2000, Weight : 0.877, Bias : -0.391, Cost : 1.373
Epoch : 3000, Weight : 0.878, Bias : -0.422, Cost : 1.372
Epoch : 4000, Weight : 0.879, Bias : -0.432, Cost : 1.372
Epoch : 5000, Weight : 0.879, Bias : -0.435, Cost : 1.372
Epoch : 6000, Weight : 0.879, Bias : -0.436, Cost : 1.372
Epoch : 7000, Weight : 0.879, Bias : -0.436, Cost : 1.372
Epoch : 8000, Weight : 0.879, Bias : -0.436, Cost : 1.372
Epoch : 9000, Weight : 0.879, Bias : -0.436, Cost : 1.372
Epoch : 10000, Weight : 0.879, Bias : -0.436, Cost : 1.372
```

## 파이토치를 이용한 선형회귀
- 가중치: torch.zeros(1, requeres_grad=True)로 0을 갖는 텐서로 초기화
    - requires_grad=True: 자동 미분 기능 사용 허용
    
- optim.SGD([weights, bias], lr=learing_rate): 확률적 경사하강법 적용
- zero_grad(): 기울기 0으로 초기화
- cost.backward(): 역전파 수행해서 매개변수 기울기 새로 계산
- optimizer.step(): 기울기 계산 결과를 최적화 함수에 반영
```py
import torch
from torch import optim


x = torch.FloatTensor([
    [1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
    [11], [12], [13], [14], [15], [16], [17], [18], [19], [20],
    [21], [22], [23], [24], [25], [26], [27], [28], [29], [30]
])
y = torch.FloatTensor([
    [0.94], [1.98], [2.88], [3.92], [3.96], [4.55], [5.64], [6.3], [7.44], [9.1],
    [8.46], [9.5], [10.67], [11.16], [14], [11.83], [14.4], [14.25], [16.2], [16.32],
    [17.46], [19.8], [18], [21.34], [22], [22.5], [24.57], [26.04], [21.6], [28.8]
])


# 하이퍼파라미터 초기화
weight = torch.zeros(1, requires_grad=True)
bias = torch.zeros(1, requires_grad=True)
learning_rate = 0.001

# 최적화 선언
optimizer = optim.SGD([weight, bias], lr=learning_rate)


for epoch in range(10000):
    hypothesis = weight * x + bias # 가중치 계산
    cost = torch.mean((hypothesis - y) ** 2) # SE
 
    optimizer.zero_grad() # 기울기 초기화
    cost.backward() # 역전파 수행 -> 매개변수 기울기 새로 계산
    optimizer.step() # 기울기 계산 경롸 최적화 함수에 반영

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch : {epoch+1:4d}, Weight : {weight.item():.3f}, Bias : {bias.item():.3f}, Cost : {cost:.3f}")
```
```
Epoch : 1000, Weight : 0.864, Bias : -0.138, Cost : 1.393
Epoch : 2000, Weight : 0.870, Bias : -0.251, Cost : 1.380
Epoch : 3000, Weight : 0.873, Bias : -0.321, Cost : 1.375
Epoch : 4000, Weight : 0.875, Bias : -0.364, Cost : 1.373
Epoch : 5000, Weight : 0.877, Bias : -0.391, Cost : 1.373
Epoch : 6000, Weight : 0.878, Bias : -0.408, Cost : 1.372
Epoch : 7000, Weight : 0.878, Bias : -0.419, Cost : 1.372
Epoch : 8000, Weight : 0.878, Bias : -0.425, Cost : 1.372
Epoch : 9000, Weight : 0.879, Bias : -0.429, Cost : 1.372
Epoch : 10000, Weight : 0.879, Bias : -0.432, Cost : 1.372
```

### 신경망 사용 버전
```py
import torch
from torch import optim, nn


x = torch.FloatTensor([
    [1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
    [11], [12], [13], [14], [15], [16], [17], [18], [19], [20],
    [21], [22], [23], [24], [25], [26], [27], [28], [29], [30]
])
y = torch.FloatTensor([
    [0.94], [1.98], [2.88], [3.92], [3.96], [4.55], [5.64], [6.3], [7.44], [9.1],
    [8.46], [9.5], [10.67], [11.16], [14], [11.83], [14.4], [14.25], [16.2], [16.32],
    [17.46], [19.8], [18], [21.34], [22], [22.5], [24.57], [26.04], [21.6], [28.8]
])


model = nn.Linear(1, 1) #입력이 1, 출력이 1인 신경망
criterion = nn.MSELoss() # 손실함수로 MSE 사용
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(10000):
    output = model(x) # 기울기 초기화
    cost = criterion(output, y) # 비용함수 계산

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch : {epoch+1:4d}, Weight : {weight.item():.3f}, Bias : {bias.item():.3f}, Cost : {cost:.3f}")
```
```
Epoch : 1000, Weight : 0.879, Bias : -0.432, Cost : 1.378
Epoch : 2000, Weight : 0.879, Bias : -0.432, Cost : 1.374
Epoch : 3000, Weight : 0.879, Bias : -0.432, Cost : 1.373
Epoch : 4000, Weight : 0.879, Bias : -0.432, Cost : 1.373
Epoch : 5000, Weight : 0.879, Bias : -0.432, Cost : 1.372
Epoch : 6000, Weight : 0.879, Bias : -0.432, Cost : 1.372
Epoch : 7000, Weight : 0.879, Bias : -0.432, Cost : 1.372
Epoch : 8000, Weight : 0.879, Bias : -0.432, Cost : 1.372
Epoch : 9000, Weight : 0.879, Bias : -0.432, Cost : 1.372
Epoch : 10000, Weight : 0.879, Bias : -0.432, Cost : 1.372
```