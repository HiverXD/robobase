import torch
import numpy as np
import torch.nn.functional as F
from diffusers import DDIMScheduler

from robobase.method.bc import BC
from robobase.models.encoder import EncoderModule
from robobase.models.fully_connected import FullyConnectedModule
from robobase.method.utils import (
    extract_from_spec,
    extract_many_from_spec,
    stack_tensor_dictionary,
    flatten_time_dim_into_channel_dim,
)

class DiT(BC):
    def __init__(
        self,
        encoder_model: EncoderModule,
        actor_model: FullyConnectedModule,
        diffusion_timesteps: int,
        eval_diffusion_timesteps: int,
        *args,
        **kwargs
    ):
        super().__init__(
            encoder_model=encoder_model,
            actor_model=actor_model,
            *args,
            **kwargs
        )
        self.diffusion_timesteps = diffusion_timesteps
        self.eval_diffusion_timesteps = eval_diffusion_timesteps
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=self.diffusion_timesteps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def build_actor(self):
        proprio_dim = self.observation_space['low_dim_state'].shape[-1]
        self.actor = self.actor_model(proprio_dim=proprio_dim).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

    def update(self, replay_iter, step: int, replay_buffer=None):
        metrics = dict()
        batch = next(replay_iter)
        batch = {k: v.to(self.device) for k, v in batch.items() if hasattr(v, 'to')}

        rgb_obs, _, _ = self.extract_pixels(batch)
        qpos, _ = self.extract_low_dim_state(batch)
        
        image_tokens = self.encoder(rgb_obs.float())

        clean_action = batch["action"]
        noise = torch.randn_like(clean_action)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (clean_action.shape[0],), device=self.device).long()
        noisy_action = self.noise_scheduler.add_noise(clean_action, noise, timesteps)

        predicted_noise = self.actor(
            image_tokens=image_tokens, 
            qpos=qpos, 
            action=noisy_action, 
            timestep=timesteps
        )

        loss = F.mse_loss(predicted_noise, noise)

        self.actor_opt.zero_grad(set_to_none=True)
        if self.encoder_opt:
            self.encoder_opt.zero_grad(set_to_none=True)
        
        loss.backward()
        
        self.actor_opt.step()
        if self.encoder_opt:
            self.encoder_opt.step()

        metrics['dit_loss'] = loss.item()
        return metrics

    def act(self, obs: dict, step: int, eval_mode: bool):
        # Correct way to process inference-time observations
        with torch.no_grad():
            qpos = flatten_time_dim_into_channel_dim(
                extract_from_spec(obs, "low_dim_state")
            )
            rgb_obs = flatten_time_dim_into_channel_dim(
                stack_tensor_dictionary(extract_many_from_spec(obs, r"rgb.*"), 1),
                has_view_axis=True,
            )
            image_tokens = self.encoder(rgb_obs.float())
            
            batch_size = image_tokens.shape[0]

            action_shape = (batch_size, 16, 16)
            noisy_action = torch.randn(action_shape, device=self.device)
            
            timesteps_to_set = self.eval_diffusion_timesteps if eval_mode else self.diffusion_timesteps
            self.noise_scheduler.set_timesteps(timesteps_to_set)

            for t in self.noise_scheduler.timesteps:
                timesteps = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
                predicted_noise = self.actor(
                    image_tokens=image_tokens, 
                    qpos=qpos, 
                    action=noisy_action, 
                    timestep=timesteps
                )
                
                noisy_action = self.noise_scheduler.step(
                    model_output=predicted_noise,
                    timestep=t,
                    sample=noisy_action
                ).prev_sample

            return noisy_action 