import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

from sklearn.cluster import KMeans
import numpy as np
import os
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
from PIL import Image
import shutil
import numpy as np
import pandas as pd
import umap
import pickle
from .panomitocluster_network import AutoencoderFeat, AutoencoderLabel
from .panomitocluster_losses import batch_all_triplet_loss, batch_hard_triplet_loss


class AutoencoderLabelTrainer:

    def __init__(self, root_dir, output_dir, n_class=7, device=None):

        os.makedirs(root_dir, exist_ok=True)
        self.root_dir = root_dir
        self.output_dir = output_dir

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        
        self.autoencoder = AutoencoderLabel(nc=1,n_class=n_class).to(self.device)
        
        
        self.l1_loss = nn.L1Loss()

        self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=1e-4)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', 
                                                            factor=0.3, patience=3, verbose=True) 
        
        self.qalpha = 1.5
        self.palpha = 1.5
        
    def load_model(self, model_path=None):
        if model_path is None:
            save_path = os.path.join(self.root_dir, f'autoencoder_latest.pth')
        else:
            save_path = model_path
        try:
            self.autoencoder.load_state_dict(torch.load(save_path),strict=False)
            print(f"Resumed training from {save_path}")
            return True
        except:
            print("Starting training from scratch.")
            return False

    def load_dataset(self, dataset, batch_size=256, shuffle=False):   
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) 

    def train(self, dataset, batch_size=16, 
                          lr=1e-4, epochs=100,
                          save_path="autoencoder_latest.pth",
                          resume=True,
                          weights=[1.0, 1.0],
                          class_weights=[1.0,1.0,1.0,1.0,1.0,1.0,1.0]):
        save_path = os.path.join(self.root_dir, f'autoencoder_latest.pth')

        l1loss = nn.L1Loss()
        class_weights =torch.tensor(class_weights,dtype=torch.float32).to(self.device)
        entropyloss = nn.CrossEntropyLoss(weight=class_weights)


        
        train_size = int(0.6 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        
        self.autoencoder.train()
    
        self.optimizer.param_groups[0]['lr'] = lr
        for epoch in range(epochs):
            total_recon_loss = 0
            total_feat_loss = 0
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Autoencoder Epoch {epoch+1}/{epochs}")
            
            for _, datas in enumerate(progress_bar):
                images = datas[0].to(self.device) 
                labels = datas[1].to(self.device)               
                self.optimizer.zero_grad()
                pred_images, logits, z = self.autoencoder(images)
                
                loss_recon =l1loss(pred_images, images)
                               
                loss_cate = entropyloss(logits, labels)
                loss = weights[0]*loss_recon+weights[1]*loss_cate
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_recon_loss += weights[0]*loss_recon.item()
                total_feat_loss += weights[1]*loss_cate.item()
                total_loss += loss.item()
                progress_bar.set_postfix({'Loss': loss.item()})
            
            avg_recon_loss = total_recon_loss / len(train_dataloader)
            avg_feat_loss = total_feat_loss / len(train_dataloader)
            avg_loss = total_loss / len(train_dataloader)

            
            self.autoencoder.eval()
            total_val_recon_loss = 0.0
            total_val_feat_loss = 0.0
            total_val_loss = 0.0
            nt = 0
            nf = 0
            with torch.no_grad():
                for val_datas in test_dataloader:
                    val_images = val_datas[0].to(self.device)
                    val_labels = val_datas[1].to(self.device)
                    pred_images, logits, z = self.autoencoder(val_images)
                    # 
                    preds = logits.argmax(dim=1)
                    nt += (preds == val_labels).float().sum()
                    nf += (preds != val_labels).float().sum()
                    val_recon_loss = nn.L1Loss()(pred_images, val_images)
                    val_cate_loss = nn.CrossEntropyLoss()(logits, val_labels)
                    val_loss = weights[0]*val_recon_loss + weights[1]*val_cate_loss
                    total_val_recon_loss += weights[0]*val_recon_loss
                    total_val_feat_loss += weights[1]*val_cate_loss
                    total_val_loss += val_loss.item()
            avg_val_recon_loss = total_val_recon_loss / len(test_dataloader)
            avg_val_feat_loss = total_val_feat_loss / len(test_dataloader)
            avg_val_loss = total_val_loss / len(test_dataloader)
            error_rate = nt/(nt+nf)
            self.autoencoder.train()

            
            torch.save(self.autoencoder.state_dict(), save_path)
            print(f"Autoencoder saved to {save_path}")

            self.scheduler.step(avg_val_loss)
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            if lr < 1e-6:
                print("Learning rate too small, stopping training.")
                break            

            print(f'Epoch [{epoch+1}/{epochs}], '
                f'Train Loss: {avg_loss:.6f}, '
                f'Train Recon Loss: {avg_recon_loss:.6f}, '
                f'Train cate Loss: {avg_feat_loss:.6f}, '
                f'Val Loss: {avg_val_loss:.6f}, ',
                f'Val Recon Loss: {avg_val_recon_loss:.6f}, ',
                f'Val cate Loss: {avg_val_feat_loss:.6f}, ',
                f'Accuracy: {error_rate*100:.2f}, ',
                f'Learning rate: {lr:.6f}')
        
        return self.autoencoder
    
    def cluster(self, image_folder, output_dir, output_csv="mito_clusters.csv", merge_map = None):
        image_paths = sorted(glob.glob(os.path.join(image_folder, "*.png")))

        all_z = []
        all_images = []
        all_labels = []
        filenames = []

        with torch.no_grad():
             for path in tqdm(image_paths, desc="Predicting clusters"):
                
                img = Image.open(path)
                img = img.resize((128,128))            
                img = np.array(img)
                img = img / 255.0
                img = np.expand_dims(np.expand_dims(img, axis=0),axis=0) # (1, C, H, W)
                tensor_img = torch.tensor(img, dtype=torch.float32)
                _, logits, z = self.autoencoder(tensor_img.to(self.device))
                labels = logits.argmax(dim=1)
                all_z.append(z) 
                all_images.append(tensor_img)
                all_labels.append(labels)
                filenames.append(os.path.basename(path))

        all_z_np = torch.cat(all_z, dim=0).cpu().numpy()
        print("Feature shape:", all_z_np.shape)
        print("Num images:", len(filenames))
        np.save(os.path.join(image_folder, "all_z_np.npy"), all_z_np)
        all_labels_np = torch.cat(all_labels, dim=0).cpu().numpy()
        all_images_np = torch.cat(all_images, dim=0).cpu().numpy()

        
        num_clusters = 12
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)  
        
        # kmeans_path = os.path.join(self.root_dir, f"init_kmeans_model_k_12.pkl")
        # with open(kmeans_path, 'rb') as f:
        #     kmeans = pickle.load(f)
        # labels_kmeans = kmeans.predict(all_z_np)
        labels_kmeans = kmeans.fit_predict(all_z_np)
       
        if merge_map != None:
            predictions = []
            category_colors = {}
            for merge_cat_id, cat in enumerate(merge_map):
                category_colors.setdefault(list(cat.keys())[0], merge_cat_id)
 
            pop_ids = []
            for label_id, label in enumerate(labels_kmeans):
                match_cat_colors = False
                for mapping_dict in merge_map:
                    
                    category, label_list = next(iter(mapping_dict.items()))
                    if label in label_list:
                        label = category
                        predictions.append(label)
                        match_cat_colors = True
                if match_cat_colors == False:
                    pop_ids.append(label_id)

            labels_kmeans = np.array([category_colors.get(label, 99) for label in predictions])
            
            indices_to_pop = sorted(pop_ids, reverse=True)
            for idx in indices_to_pop:
                
                if idx < len(image_paths):
                    image_paths.pop(idx)
                if idx < len(all_z_np):
                    all_z_np = np.delete(all_z_np, idx, axis=0)
                
        export_image_with_labels(output_dir, image_paths, labels_kmeans, labeltype="KmeansLabelRefine")

        filenames = []
        for path in tqdm(image_paths, desc="Predicting clusters"):                
            filenames.append(os.path.basename(path))


        df = pd.DataFrame({"filename": filenames, "cluster": labels_kmeans})
        output_csv_path = os.path.join(output_dir, output_csv)
        df.to_csv(output_csv_path, index=False)
        print(f"Saved predictions to {output_csv_path}")
        
        reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_jobs=4,                    
            low_memory=True,           
            n_neighbors=30,    
            min_dist=0.7,
            metric='euclidean'          
        )        
        z_2d = reducer.fit_transform(all_z_np)

        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels_kmeans, cmap='tab20', s=15, alpha=0.8)
        plt.colorbar(scatter, label='Cluster ID')
        plt.title(f'Clustering Result (K={len(np.unique(labels_kmeans))})')
        plt.savefig(os.path.join(output_dir, f'clustering_umap.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def cluster_dataset(self, epoch):
        save_dir = os.path.join(self.output_dir, f'epoch_{epoch:03d}')
        os.makedirs(save_dir, exist_ok=True)

       
        all_z = []
        all_images = []
        with torch.no_grad():
            for images, _ in self.dataloader:
                images = images.to(self.device)
                _, feat, z = self.autoencoder(images)
                all_z.append(z) 
                all_images.append(images)

        all_z_np = torch.cat(all_z, dim=0).cpu().numpy()

        kmeans = KMeans(n_clusters=12, random_state=42, n_init=10)  
        labels_kmeans = kmeans.fit_predict(all_z_np)

        
        reducer = umap.UMAP(
            n_components=2,
            random_state=42,
            n_jobs=4,                    
            low_memory=True,           
            n_neighbors=30,     
            min_dist=0.7,
            metric='euclidean'          
        )        
        z_2d = reducer.fit_transform(all_z_np)

        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=labels_kmeans, cmap='tab20', s=15, alpha=0.8)
        plt.colorbar(scatter, label='Cluster ID')
        plt.title(f'Clustering Result (K={len(np.unique(labels_kmeans))})')
        plt.savefig(os.path.join(save_dir, f'clustering_umap.png'), dpi=150, bbox_inches='tight')
        plt.close()

        
        unique_labels = np.unique(labels_kmeans)
        images_np = torch.cat(all_images, dim=0).cpu().numpy()  

        
        if images_np.shape[1] == 1:
            images_np = images_np.squeeze(1)  
        elif images_np.shape[1] == 3:
            images_np = np.transpose(images_np, (0, 2, 3, 1))  

        n_show = 36 
        for label in unique_labels:
            if label == -1:
                continue 
            idxs = np.where(labels_kmeans == label)[0]
            selected_idxs = np.random.choice(idxs, size=min(n_show, len(idxs)), replace=False)
            selected_imgs = images_np[selected_idxs]

            
            fig, axes = plt.subplots(6, 6, figsize=(12, 12))
            for i, ax in enumerate(axes.flat):
                if i < len(selected_imgs):
                    img = selected_imgs[i]
                    if img.ndim == 2:
                        ax.imshow(img, cmap='gray')
                    else:
                        ax.imshow(img)
                ax.axis('off')
            plt.suptitle(f'Cluster {label} (n={len(idxs)})', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'cluster_{label}_samples.png'), dpi=300)
            plt.close()

        save_path = os.path.join(save_dir, f'autoencoder.pth')
        
        torch.save(self.autoencoder.state_dict(), save_path)

    def init_clustering_kmeans(self, dataset, epoch=-1, num_clusters=12): 
        
        all_z = []
        with torch.no_grad():
            for images, _ in self.dataloader:
                images = images.to(self.device)
                _, logits, z = self.autoencoder(images)
                
                all_z.append(z) 

        all_z_tensor = torch.cat(all_z, dim=0)
        all_z_np = torch.cat(all_z, dim=0).cpu().numpy()

        z_for_clustering = all_z_np

        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)  
        labels_kmeans = kmeans.fit_predict(z_for_clustering)

        new_centers = np.zeros((num_clusters, all_z_np.shape[1]), dtype=np.float32)
        for i in range(num_clusters):
            cluster_points = all_z_np[labels_kmeans == i]
            if len(cluster_points) > 0:
                new_centers[i] = cluster_points.mean(axis=0)
            else:
                
                new_centers[i] = np.zeros(all_z_np.shape[1], dtype=np.float32)
        new_centers = torch.tensor(new_centers, dtype=torch.float32, device=self.device)      

        
        self.cluster_centers = torch.nn.Parameter(new_centers)

        
        self.optimizer = torch.optim.Adam([
            {"params": self.autoencoder.parameters(), "lr": 1e-4},
            {"params": [self.cluster_centers], "lr": 1e-4}
        ])

        dist = torch.sum((all_z_tensor.unsqueeze(1) - new_centers.unsqueeze(0)) ** 2, dim=2)
        q_full = 1.0 / (1.0 + dist / self.qalpha)
        q_full = q_full / torch.sum(q_full, dim=1, keepdim=True)
        q_full = torch.clamp(q_full, 1e-10, 1.0)
        p = self.target_distribution(q_full).detach()

        
        self.hard_labels = torch.argmin(dist, dim=1)

        
        current_p_entropy = -(p * torch.log(p)).sum(dim=1).mean().item()
        q_entropy_batch = -(q_full * torch.log(q_full)).sum(dim=1).mean().item()
        q_max_mean = q_full.max(dim=1)[0].mean().item()
        print(f'Init k: {num_clusters}, '
            f'P Entropy: {current_p_entropy:.6f}, '
            f'Q Entropy: {q_entropy_batch:.6f}, '
            f'Q Max Mean: {q_max_mean:.6f}, '
            f"P max mean: {p.max(dim=1)[0].mean().item():.4f}"
            )
        
        self.p_target = p
        return p,new_centers,labels_kmeans

    def update_target_distribution(self, refresh_p=False):
       
        
        all_z = []
        all_images = []
        with torch.no_grad():
            for images, _ in self.dataloader:
                images = images.to(self.device)
                _, logits, z = self.autoencoder(images)
                
                all_z.append(z) 
                all_images.append(images)

        all_z_tensor = torch.cat(all_z, dim=0)

        
        dist = torch.sum((all_z_tensor.unsqueeze(1) - self.cluster_centers.unsqueeze(0)) ** 2, dim=2)
        q_full = 1.0 / (1.0 + dist / self.qalpha)
        q_full = q_full / torch.sum(q_full, dim=1, keepdim=True)
        q_full = torch.clamp(q_full, 1e-10, 1.0)

        p = self.target_distribution(q_full).detach()
        if refresh_p:
            self.p_target = p  

        self.hard_labels = torch.argmin(dist, dim=1)
        
        current_p_entropy = -(p * torch.log(p)).sum(dim=1).mean().item()
        q_entropy_batch = -(q_full * torch.log(q_full)).sum(dim=1).mean().item()
        q_max_mean = q_full.max(dim=1)[0].mean().item()
        print(f'Updated p | P Entropy: {current_p_entropy:.6f}, '
              f'Q Entropy: {q_entropy_batch:.6f}, '
              f'Q Max Mean: {q_max_mean:.6f}, '
              f"P max mean: {p.max(dim=1)[0].mean().item():.4f}")
        
    def train_kmeans(self, epochs=100, loss_weights=[1.0,1.0,1.0,1.0]):

        for epoch in range(epochs):
          
            total_kl_loss = 0.0
            total_recon_loss = 0.0
            total_gt_triplet_loss = 0.0
            total_self_triplet_loss = 0.0
            total_loss = 0.0
            count = 0
            
            for i, (images, labels) in enumerate(tqdm(self.dataloader)):

                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                images_pred, logits, z = self.autoencoder(images)
                

                recon_loss = self.l1_loss(images, images_pred)
                total_recon_loss += recon_loss.item()

                
                dist = torch.sum((z.unsqueeze(1) - self.cluster_centers.unsqueeze(0)) ** 2, dim=2)
                q = 1.0 / (1.0 + dist/self.qalpha)
                q = q / torch.sum(q, dim=1, keepdim=True)
                q = torch.clamp(q, 1e-10, 1.0)

                start_idx = i * self.dataloader.batch_size
                end_idx = start_idx + images.size(0)
                p_batch = self.p_target[start_idx:end_idx]

               
                kl_loss = torch.sum(q * (torch.log(q) - torch.log(p_batch + 1e-10)), dim=1).mean()
                total_kl_loss += kl_loss.item()  

                gt_triplet_loss,_ = batch_all_triplet_loss(
                    labels=labels,
                    embeddings=z,
                    margin=0.2,       
                    squared=False
                )   
                self_triplet_loss,_ = batch_all_triplet_loss(
                    labels=self.hard_labels[start_idx:end_idx],
                    embeddings=z,
                    margin=0.2,      
                    squared=False
                )   

                total_gt_triplet_loss += gt_triplet_loss.item()     
                total_self_triplet_loss += self_triplet_loss.item()   

                total_loss = loss_weights[0]*recon_loss + loss_weights[1]*kl_loss + loss_weights[2]*gt_triplet_loss + loss_weights[3]*self_triplet_loss
                total_loss.backward()
                self.optimizer.step()
                
                count += 1
            
            ave_recon_loss = total_recon_loss/count
            avg_kl_loss = total_kl_loss / count
            avg_gt_triplet_loss = total_gt_triplet_loss / count            
            avg_self_triplet_loss = total_self_triplet_loss / count

            print(f'Epoch [{epoch+1}/{epochs}], '
                f'Recon Loss: {ave_recon_loss:.6f}, ' 
                f'KL Loss: {avg_kl_loss:.6f}, '                
                f'GT Triplet Loss: {avg_gt_triplet_loss:.6f}, ' 
                f'SELF Triplet Loss: {avg_self_triplet_loss:.6f}, '
                ) 
            if (epoch+1) % 5 == 0:
                self.cluster_dataset(epoch=epoch+1)                
                self.update_target_distribution(refresh_p=True)
            else:
                self.update_target_distribution(refresh_p=False)
    
    def target_distribution(self, q):
        
        q_power = q.pow(self.palpha)
        p = q_power / (q_power.sum(dim=1, keepdim=True) + 1e-8)  
        return p        



def export_image_with_labels(output_dir, image_paths, labels, labeltype="Label"):
    for path, label in tqdm(zip(image_paths, labels), total=len(image_paths), desc=labeltype+" Clustering"):
        class_dir = os.path.join(output_dir, labeltype+'_'+str(label))
        os.makedirs(class_dir, exist_ok=True)
        shutil.copyfile(path, os.path.join(class_dir, os.path.basename(path)))

