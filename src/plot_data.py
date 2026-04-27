import torch
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from torchvision import utils as vutils


def show_batch(
    img: torch.Tensor
) -> None:
    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1) # ImageNet
    std  = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1) # ImageNet

    img = img * std + mean # unnormalize
    img = img.clamp(0, 1)   # evitar valores fuera de rango
    np_img = img.numpy()
    plt.imshow(
        np.transpose(
            a=np_img,
            axes=(1, 2, 0) # Height, Width, Channels
        )
    )
    plt.axis('off')
    plt.show()


def show_features(
    conv: torch.Tensor,
    filter_index: int = 0,
    cmap: str = 'inferno'
) -> None:

    feature_layer = conv[:, filter_index, :, :]

    feature_layer = feature_layer.unsqueeze(1)

    grid = vutils.make_grid(
        feature_layer.cpu(),
        nrow=8,
        padding=2
    )

    grid = grid.clamp(0, 1)

    np_feature = grid.numpy()
    plt.imshow(
        np.transpose(
            a=np_feature,
            axes=(1, 2, 0) # Height, Width, Channels
        ),
        cmap=cmap
    )
    plt.axis('off')
    plt.show()

def plot_latent_space_3d(
    embeddings: np.ndarray,
    labels: list,
    class_names: list
) -> None:
    unique_labels = np.sort(np.unique(labels))
    t10_palette = px.colors.qualitative.T10

    # Map texts and colors
    hover_texts = [class_names[int(l)] for l in labels]
    colors_mapped = [t10_palette[int(l) % len(t10_palette)] for l in labels]

    fig = go.Figure()

    # 1. Main Scatter Plot
    fig.add_trace(go.Scatter3d(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        z=embeddings[:, 2],
        mode='markers',
        text=hover_texts,
        hoverinfo='text',
        marker=dict(size=3, color=colors_mapped, opacity=0.8),
        showlegend=False
    ))

    # 2. Add Proxy Traces for Legend
    for label in unique_labels:
        idx = int(label)
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            name=class_names[idx],
            marker=dict(color=t10_palette[idx % len(t10_palette)], size=10)
        ))

    # 3. Clean Layout Configuration
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        template="plotly_white",
        legend=dict(
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.05,
            bgcolor="rgba(255,255,255,0.5)"
        )
    )

    fig.show(config={'responsive': True})


def plot_latent_space_2d(
    embeddings_2d : np.ndarray,
    labels: list,
    class_names: list
) -> None:
    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap='tab10',
        alpha=0.6,
        s=10
    )

    # Add legend using the class names provided
    plt.legend(
        handles=scatter.legend_elements()[0],
        labels=class_names,
        loc="best",
        title="Classes"
    )

    plt.title("Latent Space Projection (UMAP 2D)", fontsize=14)
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()



def show_siamese_batch(
    dataloader: torch.utils.data.DataLoader,
    n_pairs: int = 4
) -> None:
    img1, img2, targets = next(iter(dataloader))

    # Configurar el grid: n_pares filas, 2 columnas
    fig, axes = plt.subplots(n_pairs, 2, figsize=(6, n_pairs * 2.5))

    # Helper para des-normalizar ImageNet
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i in range(n_pairs):
        # Preparar imágenes
        im1 = (img1[i] * std + mean).permute(1, 2, 0).clamp(0, 1).numpy()
        im2 = (img2[i] * std + mean).permute(1, 2, 0).clamp(0, 1).numpy()

        # Color según el target (Verde para Positivo, Rojo para Negativo)
        is_pos = targets[i] == 0.0
        color = '#2ecc71' if is_pos else '#e74c3c'
        label = "MISMA CLASE" if is_pos else "DIFERENTE"

        # Columna A
        axes[i, 0].imshow(im1)
        axes[i, 0].axis('off')

        # Columna B
        axes[i, 1].imshow(im2)
        axes[i, 1].axis('off')


        # Texto centralizado entre las dos imágenes
        fig.text(0.5, 1 - (i + 0.5) / n_pairs, f"{label} ({int(targets[i])})",
                 ha='center', va='center', color='white', weight='bold',
                 bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.3'))

    plt.tight_layout()
    plt.show()