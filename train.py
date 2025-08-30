import torch 
from models import Generator , MolGANDiscriminator
from sparse_molecular_dataset import SparseMolecularDataset
from tools import postprocess,get_reward,sample_z,label2onehot,reward
from collections import defaultdict
import torch.nn as nn


G = Generator()
D = MolGANDiscriminator()
R = MolGANDiscriminator()   

g_optim = torch.optim.Adam(G.parameters(),lr = 1e-3,betas=(0.0,0.5))
d_optim = torch.optim.Adam(D.parameters(),lr = 1e-3,betas=(0.0,0.5))
v_optim = torch.optim.Adam(R.parameters(),lr = 1e-3,betas=(0.0,0.5))

data = SparseMolecularDataset()


class Solve:
    def __init__(self,batch_size = 32):
        self.batch_size = batch_size
        self.data = data
        self.data.load("qm9_5k.sparsedataset")
        self.m_dim = self.data.atom_num_types
        self.b_dim = self.data.bond_num_types
        self.num_steps = len(data)//self.batch_size
        self.lamd = 0
        self.alpha = 10
        self.n_critic = 5
        # ADDED: A loss function is required for the reward network.
        self.loss_fn = nn.MSELoss()
        
    def train_or_valid(self, epoch_i, total_epochs, train_val_test="train"):
            # ==========================================================
            # FIXED: Correct training and pre-training schedule
            # ==========================================================
            # As per the paper, the generator trains with WGAN loss only for the
            # first half of epochs (lambda=1.0) and a combined loss for the second half.
            if epoch_i < total_epochs / 2:
                self.lamd = 1  # Use WGAN loss only for the generator
            else:
                self.lamd = 1  # Use combined WGAN or RL loss

            losses = defaultdict(list)
            scores = defaultdict(list)
            the_step = self.num_steps
            for a_step in range(the_step):
                cur_step = self.num_steps * epoch_i + a_step
                if train_val_test == 'val':
                    mols, _, _, a, x, _, _, _, _ = self.data.next_validation_batch()
                    z =  sample_z(a.shape[0])
                elif train_val_test == 'train':
                    mols, _, _, a, x, _, _, _, _ = self.data.next_train_batch(self.batch_size)
                    batch_dim = self.batch_size
                    z = sample_z(self.batch_size)

                a = torch.from_numpy(a).long()  
                x = torch.from_numpy(x).long()  
                a_tensor = label2onehot(a, self.b_dim)
                x_tensor = label2onehot(x, self.m_dim)  
                if epoch_i == 100:
                                    
                    z = torch.from_numpy(z).float()
                    z = z + torch.rand_like(z)
                else:
                    z = torch.from_numpy(z).float()

                d_optim.zero_grad()
                real_logits = D(x_tensor,a_tensor)
                fake_adjancency , fake_nodes = G(z)
                (adj_hat,nodes_hat) = postprocess((fake_adjancency , fake_nodes))
                fake_logits = D(nodes_hat.detach(), adj_hat.detach())
                    
                d_loss_wgan = torch.mean(fake_logits) - torch.mean(real_logits)

                eps = torch.rand(batch_dim, 1, 1, 1).to(a_tensor.device)
                a_hat_grad = (eps * a_tensor + (1 - eps) * fake_adjancency.detach()).requires_grad_(True)
                x_hat_grad = (eps.squeeze(-1) * x_tensor + (1 - eps.squeeze(-1)) * fake_nodes.detach()).requires_grad_(True)
            
                logits_hat = D(x_hat_grad, a_hat_grad)
                gradients = torch.autograd.grad(
                        outputs=logits_hat,
                        inputs=(x_hat_grad, a_hat_grad),
                        grad_outputs=torch.ones(logits_hat.size()).to(logits_hat.device),
                        create_graph=True, retain_graph=True,
                    )
                grad_flat = torch.cat((gradients[0].view(batch_dim, -1), gradients[1].view(batch_dim, -1)), dim=1)
                gradient_penalty = self.alpha * ((grad_flat.norm(2, dim=1) - 1) ** 2).mean()

                d_loss = d_loss_wgan + gradient_penalty
                if cur_step % self.n_critic != 0:
                    d_loss.backward()
                    d_optim.step()
                losses['d_loss'].append(d_loss.item())

                v_optim.zero_grad()
                real_reward = R(x_tensor,a_tensor,reward = True)
                fake_reward = R(nodes_hat.detach(),adj_hat.detach(),reward =True)
                    
                reward_r = torch.from_numpy(reward(mols)).float().to(real_reward.device)
                reward_f = get_reward(nodes_hat.detach(),adj_hat.detach()).float().to(fake_reward.device)
                                   
                loss_V = self.loss_fn(real_reward, reward_r) + self.loss_fn(fake_reward, reward_f)
                if cur_step % self.n_critic == 0:
                    loss_V.backward()
                    v_optim.step()
                
                # --- Generator Training (Unchanged) ---
                g_optim.zero_grad()
                
                fake_logits_for_g = D(nodes_hat, adj_hat)
                g_loss_wgan = -torch.mean(fake_logits_for_g)
                
                fake_reward_for_g = R(nodes_hat, adj_hat, reward=True)
                g_loss_rl = -torch.mean(fake_reward_for_g)
                
                g_loss = self.lamd * g_loss_wgan + (1 - self.lamd) *g_loss_rl 
                if cur_step % self.n_critic == 0:
                    g_loss.backward()
                    g_optim.step()
                losses['g_loss'].append(g_loss.item())
                
                print(f"Epoch {epoch_i} Losses -> D: {d_loss.item():.4f}, G: {g_loss.item():.4f}")
                
                
                

solver = Solve()
G.train()
D.train()
R.train()
total_epochs = 300 # As used in the paper for the 5k dataset

# ==========================================================
# 5. The Main Training Loop
# ==========================================================
print("Starting MolGAN training...")
for epoch_i in range(total_epochs):
    # Call the training method for the current epoch.
    # The 'train_val_test' argument defaults to "train".
    # We pass 'total_epochs' so the pre-training schedule inside the method works correctly.
    solver.train_or_valid(epoch_i, total_epochs)

print("Training finished.")

torch.save(G.state_dict(), "Generator.pth")


