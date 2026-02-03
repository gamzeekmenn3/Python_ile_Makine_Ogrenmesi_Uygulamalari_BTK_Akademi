'''
Bu proje, bir Q-Learning algoritması uygulamasıdır. Amaç, FrozenLake-v1 adlı 4x4'lük bir ızgara dünyasında, başlangıç noktasından (S) bitiş noktasına (G) ulaşmaktır. Ajan, deliklere (H) düşmeden güvenli buz (F) 
üzerinde yürümeyi deneme-yanılma yoluyla öğrenir. is_slippery=False olduğu için ajan hangi yöne gitmek isterse tam oraya gider (deterministik ortam).

* Kullanılan Kütüphaneler:
- Gymnasium (gym): OpenAI tarafından geliştirilen ve ajan eğitmek için kullanılan standart oyun/simülasyon ortamıdır.
- NumPy (np): Q-Tablosu (ajanın hafızası) oluşturmak ve matematiksel işlemler yapmak için kullanılır.
- Matplotlib (plt): Eğitim sonuçlarını görselleştirmek (başarı grafiği) için kullanılır.
- tqdm: (Kodda import edilmiş ama döngüde kullanılmamış) Genelde eğitim sürecini bir ilerleme çubuğu ile izlemek için kullanılır.
'''
import gymnasium as gym
import random
import numpy as np

environment = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
environment.reset()

nb_states = environment.observation_space.n
nb_actions = environment.action_space.n
qtable = np.zeros((nb_states, nb_actions))
print("Q-table:")
print(qtable)

action = environment.action_space.sample()
new_state, reward, terminated, truncated, info = environment.step(action)
done = terminated or truncated

# %%
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

environment = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
environment.reset()

nb_states = environment.observation_space.n
nb_actions = environment.action_space.n
qtable = np.zeros((nb_states, nb_actions))
print("Q-table:")
print(qtable)

episodes = 1000  
alpha = 0.5    
gamma = 0.9      
outcomes = [] 

for _ in range(episodes):
    state, _ = environment.reset()
    done = False
    outcomes.append("Failure")
    while not done:  
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])
        else:
            action = environment.action_space.sample()
        new_state, reward, terminated, truncated, info = environment.step(action)
        done = terminated or truncated
        qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])
        state = new_state
        if reward:
            outcomes[-1] = "Success"
print("Qtable after training: ")
print(qtable)

plt.bar(range(episodes), outcomes)
episodes = 1000  
nb_success = 0

for _ in range(episodes):
    state, _ = environment.reset()
    done = False
    while not done:
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])
        else:
            action = environment.action_space.sample()
        new_state, reward, terminated, truncated, info = environment.step(action)
        done = terminated or truncated
        state = new_state
        nb_success += reward
print("Success rate:", 100 * nb_success / episodes)

'''
* Kod Akışı:
1. Hazırlık ve Q-Tablosu Oluşturma: 
Ortam (Environment) başlatılır ve tüm değerleri sıfır olan bir Q-Table oluşturulur. Bu tablo, her bir durumda (karede) hangi yönün (yukarı, aşağı, sol, sağ) ne kadar "değerli" olduğunu tutan bir matristir.
2. Eğitim (Training) Fazı:
Ajan 1000 bölüm boyunca gölde yürür. Her adımda şu denklemle (Bellman Denklemi) tablosunu günceller: Q(s, a) = Q(s, a) + α [R + γ*max Q(s', a') - Q(s, a)] 
    - Epsilon-Greedy Benzeri Mantık: Eğer ajan bir kalede daha önce ödül bulmuşsa o yöne gider, bulamamışsa rastgele hareket eder.
    - Ödül: Hedefe ulaştığında 1, diğer durumlarda 0 alır.
3. Test ve Değerlendirme:
Eğitim bittikten sonra ajan "akıllanmış" olan Q-Tablosunu kullanarak 1000 kez daha oynar. Bu aşamada artık öğrenmez, sadece bildiği en iyi yolları kullanır. 
Sonuç olarak ekrana bir Başarı Oranı (Success Rate) basılır.
'''
