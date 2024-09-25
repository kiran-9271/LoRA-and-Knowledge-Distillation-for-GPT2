import torch


def train(teacher_model,student_model,train_loader, optimizer, criterion, device,mode):

    total_loss = 0.0
    total = 0
    correct = 0
    if(mode=="distil"):
        teacher_model.eval()
        student_model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            X,mask, Y = batch
            X,mask, Y = X.to(device),mask.to(device), Y.to(device)
            outputs = student_model(X,mask)
            with torch.no_grad():
                teacher_outputs = teacher_model(X,mask)
            student_out = student_model(X,mask)
            T = 2
            s_y=torch.nn.functional.softmax(teacher_outputs/T,dim=-1)
            log_s=torch.nn.functional.log_softmax(student_out/T,dim=-1)
            loss=torch.nn.functional.kl_div(log_s,s_y,reduction='batchmean')*T*T

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            prediv = torch.argmax(outputs, dim=1)
            correct += (prediv == Y).sum().item()
            total += len(Y)

        
    elif(mode=="LoRA"):
        teacher_model.train()
        
        for batch in train_loader:
            optimizer.zero_grad()
            X,mask, Y = batch
            X,mask, Y = X.to(device),mask.to(device), Y.to(device)
            outputs =teacher_model(X,mask)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            prediv = torch.argmax(outputs, dim=1)
            correct += (prediv == Y).sum().item()
            total += len(Y)


    elif(mode=="rnn"):
        student_model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            X,mask, Y = batch
            X,mask, Y = X.to(device),mask.to(device), Y.to(device)
            outputs = student_model(X,mask)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            prediv = torch.argmax(outputs, dim=1)
            correct += (prediv == Y).sum().item()
            total += Y.size(0)
    return total_loss / len(train_loader), correct/total

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    total = 0
    correct = 0
    for batch in val_loader:
        X,mask, Y = batch
        X,mask, Y = X.to(device),mask.to(device), Y.to(device)
        outputs = model(X,mask)
        loss = criterion(outputs, Y)
        total_loss += loss.item()
        prediv = torch.argmax(outputs, dim=1)
        correct += (prediv == Y).sum().item()
        total += len(Y)
    return total_loss / len(val_loader), correct / total
