import torch
import torch.nn as nn
import torch.optim as optim
from projects.social_interactions.src.models.mtcnn.utils import utils


# Define optimizer and loss function for MTCNN
mtcnn_optimizer = optim.Adam(mtcnn_model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn_model = mtcnn_model.to(device)

# Training loop for MTCNN
num_epochs = 50
for epoch in range(num_epochs):
    mtcnn_train_loss = utils.train_mtcnn_model(mtcnn_model, train_loader, mtcnn_optimizer, loss_fn, device)
    mtcnn_val_loss = utils.evaluate_mtcnn_model(mtcnn_model, val_loader, loss_fn, device)
    
    print(f"Epoch {epoch+1}/{num_epochs}, "
    f"MTCNN Train Loss: {mtcnn_train_loss:.4f}, MTCNN Val Loss: {mtcnn_val_loss:.4f}")
