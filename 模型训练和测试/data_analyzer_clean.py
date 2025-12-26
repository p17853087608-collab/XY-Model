import os
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import json

class FeatureDataLoaderClean:
    """标准模型特征数据加载器，用于加载和分析保存的稀疏矩阵数据"""
    
    def __init__(self, data_folder="数据保存"):
        self.data_folder = data_folder
        self.image_features = None
        self.metadata = None
        self.feature_stats = None
        self.training_data = None
    
    def load_features(self):
        """加载所有特征数据"""
        try:
            # 加载图像特征稀疏矩阵
            img_path = os.path.join(self.data_folder, "image_features_sparse_clean.npz")
            if os.path.exists(img_path):
                self.image_features = sparse.load_npz(img_path)
                print(f"图像特征矩阵加载成功: {self.image_features.shape}")
            
            # 加载元数据
            meta_path = os.path.join(self.data_folder, "feature_metadata_clean.csv")
            if os.path.exists(meta_path):
                self.metadata = pd.read_csv(meta_path)
                print(f"元数据加载成功: {len(self.metadata)} 条记录")
            
            # 加载特征统计
            stats_path = os.path.join(self.data_folder, "feature_statistics_clean.json")
            if os.path.exists(stats_path):
                with open(stats_path, 'r', encoding='utf-8') as f:
                    self.feature_stats = json.load(f)
                print("特征统计数据加载成功")
            
            # 加载训练数据
            training_path = os.path.join(self.data_folder, "detailed_training_data_clean.csv")
            if os.path.exists(training_path):
                self.training_data = pd.read_csv(training_path)
                print(f"详细训练数据加载成功: {len(self.training_data)} 条记录")
            
            return True
        
        except Exception as e:
            print(f"加载特征数据失败: {str(e)}")
            return False
    
    def get_dense_features(self):
        """获取密集格式的特征"""
        img_dense = self.image_features.toarray() if self.image_features is not None else None
        return img_dense
    
    def analyze_feature_distribution(self):
        """分析特征分布"""
        if self.metadata is None:
            print("元数据未加载，无法分析")
            return
        
        print("\n=== 特征分布分析 ===")
        print(f"样本总数: {len(self.metadata)}")
        print(f"标签分布: 类别0 ({(self.metadata['labels'] == 0).sum()}个), 类别1 ({(self.metadata['labels'] == 1).sum()}个)")
        
        if self.feature_stats:
            print(f"图像特征稀疏度: {self.feature_stats['image_features_sparsity']:.4f}")
            print(f"特征范围: [{self.feature_stats['feature_range'][0]:.4f}, {self.feature_stats['feature_range'][1]:.4f}]")
            print(f"特征均值: {self.feature_stats['feature_mean']:.4f}, 标准差: {self.feature_stats['feature_std']:.4f}")
        
        if self.training_data is not None:
            print(f"\n训练数据统计:")
            print(f"  总样本数: {len(self.training_data)}")
            print(f"  训练样本: {len(self.training_data[self.training_data['phase'] == 'train'])}")
            print(f"  验证样本: {len(self.training_data[self.training_data['phase'] == 'val'])}")
            print(f"  损失范围: [{self.training_data['loss'].min():.6f}, {self.training_data['loss'].max():.6f}]")
            print(f"  平均损失: {self.training_data['loss'].mean():.6f}")
            print(f"  损失标准差: {self.training_data['loss'].std():.6f}")
    
    def plot_loss_analysis(self):
        """绘制损失函数分析图"""
        if self.training_data is None:
            print("未找到详细训练数据文件")
            return
        
        df = self.training_data
        
        plt.figure(figsize=(15, 10))
        
        # 训练vs验证损失分布
        plt.subplot(2, 3, 1)
        train_loss = df[df['phase'] == 'train']['loss']
        val_loss = df[df['phase'] == 'val']['loss']
        plt.boxplot([train_loss, val_loss], labels=['Train', 'Validation'])
        plt.ylabel('Loss')
        plt.title('Train vs Validation Loss')
        
        # 损失分布直方图
        plt.subplot(2, 3, 2)
        plt.hist(df['loss'], bins=50, alpha=0.7)
        plt.xlabel('Loss')
        plt.ylabel('Frequency')
        plt.title('Loss Distribution')
        
        # 按epoch的损失变化
        plt.subplot(2, 3, 3)
        epoch_loss = df.groupby('epoch')['loss'].mean()
        plt.plot(epoch_loss.index, epoch_loss.values, 'o-')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.title('Loss by Epoch')
        plt.grid(True, alpha=0.3)
        
        # 正确vs错误的损失对比
        plt.subplot(2, 3, 4)
        correct_loss = df[df['correct'] == True]['loss']
        wrong_loss = df[df['correct'] == False]['loss']
        if len(correct_loss) > 0 and len(wrong_loss) > 0:
            plt.boxplot([correct_loss, wrong_loss], labels=['Correct', 'Wrong'])
            plt.ylabel('Loss')
            plt.title('Loss by Prediction Correctness')
        
        # 预测准确率随epoch变化
        plt.subplot(2, 3, 5)
        epoch_acc = df.groupby('epoch')['correct'].mean()
        plt.plot(epoch_acc.index, epoch_acc.values, 'o-', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy by Epoch')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        # 按batch的损失变化
        plt.subplot(2, 3, 6)
        batch_loss = df.groupby('batch')['loss'].mean()
        plt.plot(batch_loss.index[:100], batch_loss.values[:100], 'o-', alpha=0.5)  # 显示前100个batch
        plt.xlabel('Batch')
        plt.ylabel('Average Loss')
        plt.title('Loss by Batch (First 100)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_folder, "loss_analysis_clean.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    def perform_pca_analysis(self):
        """对特征进行PCA降维分析"""
        img_dense = self.get_dense_features()
        
        if img_dense is None or self.metadata is None:
            print("数据不完整，无法进行PCA分析")
            return
        
        plt.figure(figsize=(18, 6))
        
        # 图像特征PCA - 2D
        plt.subplot(1, 3, 1)
        pca_img = PCA(n_components=2)
        img_pca = pca_img.fit_transform(img_dense)
        scatter = plt.scatter(img_pca[:, 0], img_pca[:, 1], 
                           c=self.metadata['labels'], cmap='viridis', alpha=0.6)
        plt.xlabel(f'PC1 ({pca_img.explained_variance_ratio_[0]:.3f})')
        plt.ylabel(f'PC2 ({pca_img.explained_variance_ratio_[1]:.3f})')
        plt.title('Image Features PCA (2D)')
        plt.colorbar(scatter, label='Labels')
        
        # 图像特征PCA - 3D
        plt.subplot(1, 3, 2, projection='3d')
        pca_img_3d = PCA(n_components=3)
        img_pca_3d = pca_img_3d.fit_transform(img_dense)
        scatter_3d = plt.scatter(img_pca_3d[:, 0], img_pca_3d[:, 1], img_pca_3d[:, 2],
                              c=self.metadata['labels'], cmap='viridis', alpha=0.6)
        plt.xlabel(f'PC1 ({pca_img_3d.explained_variance_ratio_[0]:.3f})')
        plt.ylabel(f'PC2 ({pca_img_3d.explained_variance_ratio_[1]:.3f})')
        plt.zlabel(f'PC3 ({pca_img_3d.explained_variance_ratio_[2]:.3f})')
        plt.title('Image Features PCA (3D)')
        
        # 解释方差比例
        plt.subplot(1, 3, 3)
        pca_full = PCA().fit(img_dense)
        cumsum_ratio = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = range(1, min(50, len(cumsum_ratio) + 1))
        plt.plot(n_components, cumsum_ratio[:len(n_components)], 'o-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Explained Variance by Components')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='90% variance')
        plt.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95% variance')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_folder, "pca_analysis_clean.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    def perform_tsne_analysis(self):
        """使用t-SNE进行非线性降维可视化"""
        img_dense = self.get_dense_features()
        
        if img_dense is None or self.metadata is None:
            print("数据不完整，无法进行t-SNE分析")
            return
        
        print("正在进行t-SNE降维（这可能需要一些时间）...")
        
        # t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_result = tsne.fit_transform(img_dense)
        
        plt.figure(figsize=(12, 5))
        
        # t-SNE可视化
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                           c=self.metadata['labels'], cmap='viridis', alpha=0.6)
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('t-SNE Visualization')
        plt.colorbar(scatter, label='Labels')
        
        # 对比PCA和t-SNE
        plt.subplot(1, 2, 2)
        pca_2d = PCA(n_components=2).fit_transform(img_dense)
        plt.scatter(pca_2d[:, 0], pca_2d[:, 1], 
                   c=self.metadata['labels'], cmap='plasma', alpha=0.4, marker='x', s=30)
        plt.xlabel('PCA 1')
        plt.ylabel('PCA 2')
        plt.title('PCA for Comparison')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_folder, "tsne_analysis_clean.png"), dpi=300, bbox_inches='tight')
        plt.show()
    
    def cluster_analysis(self, n_clusters=3):
        """聚类分析"""
        img_dense = self.get_dense_features()
        
        if img_dense is None:
            print("图像特征未加载，无法进行聚类分析")
            return
        
        # K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(img_dense)
        
        # 分析聚类结果
        print(f"\n=== 聚类分析 (k={n_clusters}) ===")
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_labels_true = self.metadata['labels'][cluster_mask]
            
            print(f"\n聚类 {i}:")
            print(f"  样本数: {cluster_mask.sum()}")
            print(f"  类别0数量: {(cluster_labels_true == 0).sum()}")
            print(f"  类别1数量: {(cluster_labels_true == 1).sum()}")
            print(f"  类别1比例: {(cluster_labels_true == 1).sum() / cluster_mask.sum():.3f}")
        
        # 可视化聚类结果
        plt.figure(figsize=(15, 5))
        
        # PCA投影下的聚类结果
        plt.subplot(1, 3, 1)
        pca = PCA(n_components=2)
        features_pca = pca.fit_transform(img_dense)
        scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], 
                           c=cluster_labels, cmap='tab10', alpha=0.6)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
        plt.title(f'K-means Clustering (k={n_clusters})')
        plt.colorbar(scatter, label='Cluster')
        
        # 真实标签vs聚类结果
        plt.subplot(1, 3, 2)
        scatter_true = plt.scatter(features_pca[:, 0], features_pca[:, 1], 
                              c=self.metadata['labels'], cmap='viridis', alpha=0.3, marker='s', s=50)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
        plt.title('True Labels (Background) vs Clusters')
        plt.colorbar(scatter, label='True Labels')
        
        # 聚类中心可视化
        plt.subplot(1, 3, 3)
        centers_pca = pca.transform(kmeans.cluster_centers_)
        plt.scatter(features_pca[:, 0], features_pca[:, 1], 
                   c=cluster_labels, cmap='tab10', alpha=0.4)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Cluster Centers')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f})')
        plt.title('Cluster Centers')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_folder, "cluster_analysis_clean.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        return cluster_labels
    
    def save_analysis_report(self):
        """保存分析报告"""
        report = {
            'data_summary': {
                'total_samples': len(self.metadata) if self.metadata is not None else 0,
                'feature_shape': list(self.image_features.shape) if self.image_features is not None else None,
                'feature_sparsity': self.feature_stats['image_features_sparsity'] if self.feature_stats else None,
                'label_distribution': {
                    'class_0': int((self.metadata['labels'] == 0).sum()) if self.metadata is not None else 0,
                    'class_1': int((self.metadata['labels'] == 1).sum()) if self.metadata is not None else 0
                } if self.metadata is not None else None
            },
            'training_summary': {
                'total_training_samples': len(self.training_data) if self.training_data is not None else 0,
                'train_samples': len(self.training_data[self.training_data['phase'] == 'train']) if self.training_data is not None else 0,
                'val_samples': len(self.training_data[self.training_data['phase'] == 'val']) if self.training_data is not None else 0,
                'min_loss': float(self.training_data['loss'].min()) if self.training_data is not None else None,
                'max_loss': float(self.training_data['loss'].max()) if self.training_data is not None else None,
                'avg_loss': float(self.training_data['loss'].mean()) if self.training_data is not None else None
            } if self.training_data is not None else None,
            'feature_statistics': self.feature_stats,
            'files_generated': [
                "loss_analysis_clean.png",
                "pca_analysis_clean.png", 
                "tsne_analysis_clean.png",
                "cluster_analysis_clean.png",
                "feature_analysis_report_clean.json"
            ]
        }
        
        report_path = os.path.join(self.data_folder, "feature_analysis_report_clean.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n分析报告已保存至: {report_path}")
        return report

def main():
    """主函数，执行完整的数据分析流程"""
    # 创建数据加载器
    loader = FeatureDataLoaderClean("数据保存")
    
    # 加载数据
    if not loader.load_features():
        print("数据加载失败，请先运行训练脚本")
        return
    
    # 分析特征分布
    loader.analyze_feature_distribution()
    
    # 绘制损失分析图
    loader.plot_loss_analysis()
    
    # 执行PCA分析
    loader.perform_pca_analysis()
    
    # 执行t-SNE分析
    loader.perform_tsne_analysis()
    
    # 执行聚类分析
    loader.cluster_analysis(n_clusters=3)
    
    # 保存分析报告
    report = loader.save_analysis_report()
    
    print("\n=== 分析完成 ===")
    print("生成的文件:")
    for file in report['files_generated']:
        print(f"  - {file}")

if __name__ == "__main__":
    main()