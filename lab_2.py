import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import random

# Определение размеров входных и выходных данных
input_size = 28 * 28  # Размер изображения (28x28 пикселей)
output_size = 10  # Количество классов (цифры от 0 до 9)

# Описание алгоритма обучения
print("Алгоритм обучения: Однослойный персептрон, метод стохастического градиентного спуска")
print("")

# Загрузка данных MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Определение архитектуры нейронной сети
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_size, output_size)  # Входной слой: 28*28 пикселей, выходной слой: 10 классов

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        return x

# Инициализация нейронной сети
net = NeuralNetwork()

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Определение критериев останова
epsilon_threshold = 0.01  # Пороговое значение для критерия останова по значению ошибки ε
max_epochs = 10  # Лимит количества эпох
convergence_epochs = 3  # Количество эпох, после которых общая ошибка считается неизменной

# Обучение нейронной сети
print("\nОбучение нейронной сети:")
print("Эпоха\tLoss")
for epoch in range(max_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    average_loss = running_loss / len(trainloader)
    print(f"{epoch + 1}\t{average_loss}")

    # Проверка критериев останова
    if average_loss < epsilon_threshold:
        print(f"Критерий останова: значение ошибки ε достигло порогового значения ({epsilon_threshold})")
        break
    
    if epoch > convergence_epochs:
        recent_losses = [running_loss / len(trainloader) for i in range(convergence_epochs)]
        if max(recent_losses) - min(recent_losses) < 0.001:
            print("Критерий останова: значение общей ошибки изменяется незначительно на протяжении нескольких эпох")
            break

# Оценка точности на тестовом наборе и тестирование на случайном изображении
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"\nТочность на тестовом наборе:\n{100 * correct / total}%")

# Тестирование на случайном изображении
print("\nТестирование на случайном изображении:")
print("Истинная метка\tПредсказанная метка")
random_index = random.randint(0, len(testset))
image, label = testset[random_index]
output = net(image.unsqueeze(0))  # Размерность изображения: (1, 1, 28, 28)
_, predicted = torch.max(output, 1)
print(f"{label}\t\t{predicted.item()}")
