import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)

episode = 0
steps_per_episode = []

for _ in range(1000):
    env.render()

    cart_pos, cart_vel, pole_angle, pole_vel = observation

    # 🧠 固定策略：根據角度與角速度控制方向
    action = 1 if pole_angle + 0.5 * pole_vel > 0 else 0

    observation, reward, terminated, truncated, info = env.step(action)

    # 計算每次撐了幾步
    if 'steps' not in info:
        info['steps'] = 0
    info['steps'] += 1

    if terminated or truncated:
        episode += 1
        steps_per_episode.append(info['steps'])
        print(f"[Episode {episode}] 撐了 {info['steps']} 步")
        observation, info = env.reset()

env.close()

# 最後統計
print("\n✅ 撐的步數總結：")
for i, steps in enumerate(steps_per_episode):
    print(f"第 {i+1} 回合：{steps} 步")
