'''
Bu proje, bir taksi sürücüsünü (ajanı), 5x5 boyutundaki bir ızgara üzerinde yolcuyu doğru yerden alıp doğru hedefe bırakması için eğitmeyi amaçlar. Taksi, yanlış hareketlerde veya hatalı yolcu alma/bırakma 
işlemlerinde ceza puanı alırken, görevi başarıyla tamamladığında ödül kazanır. Amaç, ajanın maksimum toplam ödülü alacak en kısa yolu öğrenmesidir.

* Kullanılan Kütüphaneler
- gymnasium (gym): OpenAI tarafından geliştirilen, takviyeli öğrenme algoritmaları geliştirmek ve karşılaştırmak için kullanılan standart araç takımıdır.
- numpy: Q-Tablosunu (matris) oluşturmak ve yüksek performanslı matematiksel hesaplamalar yapmak için kullanılır.
- random: Ajanın yeni yollar keşfetmesi (Exploration) için rastgele seçimler yapmasını sağlar.
- tqdm: Eğitim sürecindeki döngünün ilerlemesini görsel bir bar ile takip etmemize yarar.
'''
import gymnasium as gym
import numpy as np
import random
from tqdm import tqdm

env = gym.make("Taxi-v3", render_mode="ansi")
env.reset()
print(env.render())
# 0: South  1: North  2: East  3: West  4: Pickup 5: Dropoff

action_space = env.action_space.n
state_space = env.observation_space.n
q_table = np.zeros((state_space, action_space))

alpha = 0.1
gamma = 0.6
epsilon = 0.1

for i in tqdm(range(1, 100001)):
    state, _ = env.reset()
    done = False
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else: # exploit
            action = np.argmax(q_table[state])
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
        state = next_state
print("Training finished.")

total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state, _ = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False
    while not done:
        action = np.argmax(q_table[state])
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state
        if reward == -10:
            penalties += 1
        epochs += 1
    total_penalties += penalties
    total_epochs += epochs
print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")
'''
* Kod Akışı:
A. Hazırlık ve Parametreler:
Önce ortam oluşturulur ve tüm durum-eylem çiftlerinin değerlerini saklayacağımız Q-Tablosu (500x6 boyutunda) sıfırlarla başlatılır. Ardından öğrenme hızı (α), gelecek ödüllerin önemi (γ) ve keşif oranı (ε) gibi 
hiperparametreler tanımlanır.
B. Eğitim (Training): 100.000 bölümlük bir döngüde ajan şu adımları izler:
    1. Epsilon-Greedy Stratejisi: Ajan %10 ihtimalle rastgele bir hareket yapar (keşif), %90 ihtimalle bildiği en iyi hareketi seçer (sömürü).
    2. Q-Update (Bellman Denklemi): Ajan yaptığı hareket sonucu bir ödül alır ve Q-tablosundaki değerini şu formülle günceller: Q(s, a) <- Q(s, a) + α[R + γ max Q(s', a') - Q(s, a)]
    Bu sayede ajan, hangi durumda hangi eylemin daha karlı olduğunu öğrenir.
C. Test ve Değerlendirme:
Eğitim bittikten sonra ajan 100 bölüm boyunca test edilir. Bu aşamada artık rastgele hareket yapmaz; sadece öğrendiği en iyi hamleleri (argmax) uygular. Sonuç olarak taksinin hedefi kaç hamlede tamamladığı ve 
ne kadar ceza aldığı ekrana yazdırılır.
'''
