import numpy as np
from typing import SupportsFloat
from functools import singledispatchmethod


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        size: int,
        num_envs: int = 1,
        ignore_obs_next: bool = True
    ):
        self.max_size = int(size)
        self.num_envs = num_envs
        self.ignore_obs_next = ignore_obs_next

        self.obs = np.zeros(shape=(self.num_envs, self.max_size, state_dim), dtype=np.float32)
        if not self.ignore_obs_next:
            self.obs_next = np.zeros_like(self.obs)

        self.rew = np.zeros(shape=(self.num_envs, self.max_size, 1), dtype=np.float32)

        self.act = np.zeros(shape=(self.num_envs, self.max_size, action_dim), dtype=np.float32)
        self.act_log_prob = np.zeros_like(self.act)

        self.terminated = np.zeros(shape=(self.num_envs, self.max_size, 1), dtype=np.bool_)
        self.truncated = np.zeros_like(self.terminated, dtype=np.bool_)

        self.done = np.zeros_like(self.truncated, dtype=np.bool_)

        self.__index = np.zeros(shape=(self.num_envs, ), dtype=np.int64)
        self.__last_index = np.zeros_like(self.__index)
        self.size = np.zeros_like(self.__index)
    
    def reset(self) -> None:
        self.obs[:] = 0
        if not self.ignore_obs_next:
            self.obs_next[:] = 0
        
        self.rew[:] = 0
        self.act[:] = 0
        self.act_log_prob[:] = 0
        self.terminated[:] = False
        self.truncated[:] = False
        self.done[:] = False

        self.__index[:] = 0
        self.__last_index[:] = 0
        self.size[:] = 0

    def __get_index(self, env_indices: np.ndarray | None = None) -> np.ndarray:
        self.__last_index[:] = self.__index
        if env_indices is not None:
            self.__index[env_indices] = (self.__index[env_indices]+1) % self.max_size
            self.size[env_indices] = np.minimum(self.size[env_indices] + 1, self.max_size)
            return self.__last_index[env_indices]
        else:
            self.__index = (self.__index+1) % self.max_size
            self.size = np.minimum(self.size + 1, self.max_size)
            return self.__last_index

    @singledispatchmethod
    def add(self, rew: SupportsFloat | np.ndarray, terminated: bool | np.ndarray, truncated: bool | np.ndarray, obs: np.ndarray, act: np.ndarray, act_log_prob: np.ndarray, obs_next: np.ndarray | None = None, env_indices: np.ndarray | None = None):
        print(type(rew))
        print(type(terminated))
        print(type(truncated))
        print(type(obs))
        print(type(act))
        print(type(act_log_prob))
        print(type(obs_next))
        print(type(env_indices))
        raise NotImplementedError
    
    @add.register
    def _(self, rew: SupportsFloat, terminated: bool, truncated: bool, obs: np.ndarray, act: np.ndarray, act_log_prob: np.ndarray,  obs_next: np.ndarray | None = None, env_indices: np.ndarray | None = None):
        ptr = self.__get_index(env_indices)
        if env_indices is None:
            env_indices = np.arange(self.num_envs)
        
        self.obs[env_indices, ptr, ...] = obs
        self.act[env_indices, ptr, ...] = act
        self.act_log_prob[env_indices, ptr, ...] = act_log_prob
        self.rew[env_indices, ptr, ...] = rew
        self.terminated[env_indices, ptr, ...] = terminated
        self.truncated[env_indices, ptr, ...] = truncated
        self.done[env_indices, ptr, ...] = terminated | truncated
        if not self.ignore_obs_next:
            assert obs_next is not None
            self.obs_next[env_indices, ptr, ...] = obs_next

    @add.register
    def _(self, rew: np.ndarray, terminated: np.ndarray, truncated: np.ndarray, obs: np.ndarray, act: np.ndarray, act_log_prob: np.ndarray, obs_next: np.ndarray | None = None, env_indices: np.ndarray | None = None):
        ptr = self.__get_index(env_indices)
        if env_indices is None:
            env_indices = np.arange(self.num_envs)
        
        self.obs[env_indices, ptr, ...] = obs
        self.act[env_indices, ptr, ...] = act
        self.act_log_prob[env_indices, ptr, ...] = act_log_prob
        self.rew[env_indices, ptr, ...] = np.expand_dims(rew, axis=1)
        self.terminated[env_indices, ptr, ...] = np.expand_dims(terminated, axis=1)
        self.truncated[env_indices, ptr, ...] = np.expand_dims(truncated, axis=1)
        self.done[env_indices, ptr, ...] = np.expand_dims(np.logical_or(terminated, truncated), axis=1)
        if not self.ignore_obs_next:
            assert obs_next is not None
            self.obs_next[env_indices, ptr, ...] = obs_next
    
    # def sample(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #     done = self.done.copy()
    #     done[:, self.__last_index] = True
    #     if not self.ignore_obs_next:
    #         return self.obs, self.act, self.act_log_prob, self.obs_next, self.rew, done
    #     else:
    #         return self.obs, self.act, self.act_log_prob, np.roll(self.obs, -1, axis=1), self.rew, done
    
    def sample(self):
        indices = np.expand_dims(np.arange(self.max_size), axis=0)      # (1, max_size)
        size = np.expand_dims(self.size, axis=1)                      # (num_envs, 1)
        valid_mask = indices < size                                     # (num_envs, max_size)

        current_indices = np.expand_dims(self.__index, axis=1)
        reordered_indices = (indices+current_indices) % self.max_size   # 计算重排后的索引
        # 对于未满的缓冲区，使用原始索引；对于满的缓冲区，使用重排索引
        final_indices = np.where(
            np.expand_dims(self.size==self.max_size, axis=1), 
            reordered_indices, 
            np.broadcast_to(indices, (self.num_envs, self.max_size))
        )

        # 创建环境索引和位置索引的网格
        env_indices = np.arange(self.num_envs)[:, None]                             # (num_envs, 1)
        env_indices = np.broadcast_to(env_indices, (self.num_envs, self.max_size))  # (num_envs, max_size)

        # 计算有效数据索引
        valid_env_idx = env_indices[valid_mask]
        valid_pos_idx = final_indices[valid_mask]

        # 批量提取数据
        obs = self.obs[valid_env_idx, valid_pos_idx]                                # (total_valid_steps, state_dim)
        act = self.act[valid_env_idx, valid_pos_idx]                                # (total_valid_steps, action_dim)
        act_log_prob = self.act_log_prob[valid_env_idx, valid_pos_idx]              # (total_valid_steps, action_dim)
        rew = self.rew[valid_env_idx, valid_pos_idx]                                # (total_valid_steps, 1)
        done = self.done[valid_env_idx, valid_pos_idx]                              # (total_valid_steps, 1)
        # obs_next
        if not self.ignore_obs_next:
            obs_next = self.obs_next[valid_env_idx, valid_pos_idx]
        else:
            obs_next = np.roll(self.obs, -1, axis=1)
            obs_next = obs_next[valid_env_idx, valid_pos_idx]
        
        # 将每个环境序列的最后一个step标记为done
        cumsum_sizes = np.cumsum(self.size[self.size > 0])  # 只考虑有数据的环境
        if len(cumsum_sizes) > 0:
            last_step_indices = cumsum_sizes - 1  # 每个环境的最后一步索引
            done[last_step_indices] = True
        
        return obs, act, act_log_prob, obs_next, rew, done
