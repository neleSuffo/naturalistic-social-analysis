import torch


def train_mtcnn_model(model, dataloader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    for images, bboxes in dataloader:
        images = images.to(device)
        bboxes = bboxes.to(device)

        optimizer.zero_grad()
        
        outputs = model(images)
        loss = loss_fn(outputs, bboxes)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)


def evaluate_mtcnn_model(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, bboxes in dataloader:
            images = images.to(device)
            bboxes = bboxes.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, bboxes)
            
            running_loss += loss.item()
    
    return running_loss / len(dataloader)
