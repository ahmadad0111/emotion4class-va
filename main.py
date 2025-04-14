from fastapi import FastAPI
import torch
import numpy as np
from buffer import Buffer
from models import EEGNet1D ,PredictorNet # Import your model
import torch.nn as nn
import copy
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import BackgroundTasks



app = FastAPI()

# Use relative path for model file
model_path = "./best_model_EEG_32.pth"

model_path_ecg = "./best_model_ECG.pth"


# Load the trained model
model = EEGNet1D(in_channels=32)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"))) 
model.eval()  


# load ECG model

model_ECG = EEGNet1D(in_channels=2)
model_ECG.load_state_dict(torch.load(model_path_ecg, map_location=torch.device("cpu")))  
model_ECG.eval()  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model_ECG.to(device)


# load gen data samples
eeg_rep_data = torch.load("rep_buffer_eeg.pt")
ecg_rep_data = torch.load("rep_buffer_ecg.pt")

eeg_gen_samples = []
eeg_gen_labels = []

ecg_gen_samples = []
ecg_gen_labels = []

eeg_incoming_buffer = []
ecg_incoming_buffer = []


for class_sample in eeg_rep_data.values():
    for sample in class_sample:
        eeg_gen_samples.append(sample['data'])
        eeg_gen_labels.append(sample['label'])

for class_sample in ecg_rep_data.values():
    for sample in class_sample:
        ecg_gen_samples.append(sample['data'])
        ecg_gen_labels.append(sample['label'])

eeg_gen_samples = torch.stack(eeg_gen_samples)
ecg_gen_samples = torch.stack(ecg_gen_samples)
eeg_gen_labels = torch.tensor(eeg_gen_labels)
ecg_gen_labels = torch.tensor(ecg_gen_labels)


# # Initialize buffers
# eeg_gen_buffer = Buffer(buffer_size=2000,device=device)
# ecg_gen_buffer = Buffer(buffer_size=2000,device=device)

# # Fill buffers with generated samples
# eeg_gen_buffer.add_data(eeg_gen_samples, eeg_gen_labels)
# ecg_gen_buffer.add_data(ecg_gen_samples, ecg_gen_labels)

# print("Buffer filled with generated samples")


eeg_adapt_buffer = Buffer(buffer_size=1000,device=device)
ecg_adapt_buffer = Buffer(buffer_size=1000,device=device)

def cpc_loss(z_pred, z_true, temperature=0.1):
    similarity_matrix = torch.mm(z_pred, z_true.T)  # (batch_size, batch_size)
    similarity_matrix /= temperature

    # Create targets (diagonal is positive pair)
    batch_size = z_pred.size(0)
    targets = torch.arange(batch_size).to(z_pred.device)

    # Compute cross-entropy loss
    loss = F.cross_entropy(similarity_matrix, targets)
    return loss

def cpc_training(model,device,data):

    predictor_net = PredictorNet(input_dim=64, hidden_dim=32)
    predictor_net = predictor_net.to(device=device)    

    optimizer = torch.optim.Adam( list(model.parameters()) + list(predictor_net.parameters()), lr=1e-4, weight_decay=1e-5)

    for step in range(10):

        optimizer.zero_grad()
        output,z = model(data)
        
        z = F.normalize(z, p=2, dim=1)  

        z1, z2 = torch.chunk(z, chunks=2, dim=1)  # Split along temporal dimension

        z_pred = predictor_net(z1)

        # Compute CPC loss
        loss = cpc_loss(z_pred, z2)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        #print(f"Step {step + 1}, Loss: {loss.item()}")
    
    return model

def pseudo_label_alignment(model0, batch_data, buf_data, batch_labels, buf_labels, model, device='cuda'):
    model.eval()  # Set model to evaluation mode
    entropy_th = 0.5
    buf_data = buf_data.to(device)
    buf_labels = buf_labels.to(device)

    # Step 1: Compute centroids for each class in the memory buffer
    unique_labels = torch.unique(buf_labels)
    centroids = {}

    with torch.no_grad():
        # Get embeddings for all data in memory buffer
        _, buf_emb = model(buf_data)
        buf_emb = F.normalize(buf_emb, p=2, dim=1)

        # Calculate centroids (mean embeddings) for each class
        for label in unique_labels:
            class_data = buf_emb[buf_labels == label]
            centroids[label] = class_data.mean(dim=0)

    with torch.no_grad():
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        # Step 2: Extract embeddings for the validation batch
        _, batch_emb = model(batch_data)
        batch_emb = F.normalize(batch_emb, p=2, dim=1)

        # Step 3: Compute cosine similarity between validation samples and centroids
        sim_matrix = cosine_similarity(batch_emb.cpu().detach().numpy(), 
                                        torch.stack(list(centroids.values())).cpu().detach().numpy())

        # Step 4: For each sample, assign the label of the most similar centroid
        pseudo_labels = []
        for i in range(sim_matrix.shape[0]):
            most_similar_class_idx = np.argmax(sim_matrix[i])  # Find the most similar class centroid
            pseudo_label = unique_labels[most_similar_class_idx]  # Assign the class label
            pseudo_labels.append(pseudo_label)

        pseudo_labels = torch.tensor(pseudo_labels).to(device)

        # Store the aligned data and pseudo labels for later usage
        aligned_data = batch_data
        aligned_labels = pseudo_labels

    # concat the buf data and aligned data
    aligned_data_buf_data = torch.cat([buf_data, aligned_data], dim=0)
    aligned_labels_buf_labels = torch.cat([buf_labels, aligned_labels], dim=0)
    _, outputs = model0(aligned_data_buf_data)

    logits = F.softmax(outputs/ 10, dim=1)  
    entropies = -torch.sum(logits * torch.log(logits + 1e-9), dim=1)

    selected_indices = []

    # Step 5: Process each class separately and select the top samples based on entropy
    for label in unique_labels:
        # Get the indices of samples corresponding to this class
        class_indices = (aligned_labels_buf_labels == label).nonzero().squeeze()

        # Get the entropies for this class
        class_entropies = entropies[class_indices]

        # Sort the indices based on entropy values (ascending order)
        sorted_class_indices = class_indices[torch.argsort(class_entropies, descending=False)]

        # Select a balanced number of samples (up to 200 in total)
        max_samples_per_class = 200 // len(unique_labels)  # Adjust based on the number of unique labels
        if len(sorted_class_indices) > max_samples_per_class:
            selected_indices.append(sorted_class_indices[:max_samples_per_class])
        else:
            selected_indices.append(sorted_class_indices)

    # Flatten the list of selected indices and ensure they are balanced
    selected_indices = torch.cat(selected_indices)

    # Select the data and labels corresponding to the selected indices
    data_to_add = aligned_data_buf_data[selected_indices]
    labels_to_add = aligned_labels_buf_labels[selected_indices]

    return aligned_data, aligned_labels, data_to_add, labels_to_add

def fast_adapt_with_buffers(model, data, buffer, gen_samples, gen_targets, epochs=1, lr=1e-4,device=device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    data = data.to(device)
    #target = target.to(device)

    if not buffer.is_empty():
        #print("Buffer is not empty")
        buff_data, buff_labels = buffer.get_all_data()
        data_comb = torch.cat([data, buff_data], dim=0)
        personalized_model = cpc_training(copy.deepcopy(model),device,data)
    else:
        #print("Buffer is empty")
        personalized_model = cpc_training(copy.deepcopy(model),device,data)

    _,embeddings = personalized_model(data)

    embeddings = F.normalize(embeddings, p=2, dim=1)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10, max_iter=1000)
    pseudo_labels = kmeans.fit_predict(embeddings.cpu().detach().numpy())



    # check if memory buffer is empty
    if buffer.is_empty():
        #print("Buffer is empty adding data")
        buffer.add_data(examples=data, labels=torch.tensor(pseudo_labels).to(device))

        data_cluster = data
        pseudo_labels_cluster = torch.tensor(pseudo_labels).to(device)
    else:
        # get all the data from the buffer
        buf_data, buf_labels = buffer.get_all_data()

        pseudo_labels = torch.tensor(pseudo_labels).to(device)

        data_cluster,pseudo_labels_cluster, data_to_add, labels_to_add = pseudo_label_alignment(copy.deepcopy(model),data, buf_data, pseudo_labels, buf_labels, personalized_model, device=device)

        # update the buffer
        #buffer.add_data(examples=data_cluster, labels=pseudo_labels_cluster)
        buffer.add_data(examples=data_to_add, labels=labels_to_add)

    model.train()
    for steps in range(1):
        buf_data, buf_labels = buffer.get_data(32, transform=None)        
        
        data_comb = torch.cat([data_cluster, buf_data], dim=0)
        label_comb = torch.cat([pseudo_labels_cluster, buf_labels], dim=0)

        gen_samples = gen_samples.to(device)
        gen_targets = gen_targets.to(device)

        data_comb = torch.cat([data_comb, gen_samples], dim=0)
        label_comb = torch.cat([label_comb, gen_targets], dim=0)

        train_data = TensorDataset(data_comb, label_comb)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        for data, label in train_loader:
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output,_ = model(data)
            loss = criterion(output, label.long())
            loss.backward()
            optimizer.step()
    
    print("Model adapted with new data")
    # print data size
    print("Data size:", len(train_loader))

    model.eval()

# Inference Function
def infer(eeg_data = None, ecg_data=None):
    if eeg_data is not None:

        eeg_tensor = torch.tensor(eeg_data).unsqueeze(0).float()
        eeg_tensor = eeg_tensor.to(device)
        print("EEG Tensor Shape:", eeg_tensor.shape)

        with torch.no_grad():
            output_eeg,_ = model(eeg_tensor)
    else:
        output_eeg = None
    # Check if ECG data is provided

    if ecg_data is not None:
        ecg_tensor = torch.tensor(ecg_data).unsqueeze(0).float()
        ecg_tensor = ecg_tensor.to(device)
        print("ECG Tensor Shape:", ecg_tensor.shape)

        # Pass ECG data through the ECG model
        with torch.no_grad():
            output_ecg,_ = model_ECG(ecg_tensor)
    else:
        output_ecg = None


    # average the outputs if both EEG and ECG data are provided
    if output_eeg is not None and output_ecg is not None:
        output = (output_eeg + output_ecg) / 2
    elif output_ecg is not None:
        output = output_ecg
    elif output_eeg is not None:
        output = output_eeg
    else:
        output = None


    # Convert to dictionary with class probabilities [HVHA, HVLA, LVLA, LVHA] ["excitement", "relaxation", "depression", "stress"]
    classes =  ["excitement", "relaxation", "depression", "stress"]  
    probabilities = torch.softmax(output, dim=1).squeeze().tolist()
    model_output = dict(zip(classes, probabilities))

    # Get max prediction
    max_category = max(model_output, key=model_output.get)
    
    return {"prediction": max_category, "probabilities": model_output}

# @app.post("/predict/")
# async def predict(data: dict):
#     eeg_data = data["eeg_data"]  # Input EEG data
#     ecg_data = data.get("ecg_data")

#     if eeg_data is not None:
#         eeg_incoming_buffer.append(torch.tensor(eeg_data).float())
#     if ecg_data is not None:
#         ecg_incoming_buffer.append(torch.tensor(ecg_data).float())


#     if len(eeg_incoming_buffer) > 16:
#         print("Adapting model with new EEG samples...")
#         eeg_batch = torch.stack(eeg_incoming_buffer[:16])
#         fast_adapt_with_buffers(model, eeg_batch, eeg_adapt_buffer, eeg_gen_samples, eeg_gen_labels,device=device)
#         del eeg_incoming_buffer[:16]
#     if len(ecg_incoming_buffer) > 16:
#         print("Adapting model with new ECG samples...")
#         ecg_batch = torch.stack(ecg_incoming_buffer[:16])
#         fast_adapt_with_buffers(model_ECG, ecg_batch, ecg_adapt_buffer, ecg_gen_samples, ecg_gen_labels,device=device)
#         del ecg_incoming_buffer[:16]


#     result = infer(eeg_data, ecg_data)

#     return result

def adapt_model(model, batch, buffer, gen_samples, gen_labels, device):
    print("Running background adaptation...")

    fast_adapt_with_buffers(model, batch, buffer, gen_samples, gen_labels, device=device)


@app.post("/predict/")
async def predict(data: dict, background_tasks: BackgroundTasks):
    eeg_data = data["eeg_data"]  # Input EEG data
    ecg_data = data.get("ecg_data")

    if eeg_data is not None:
        eeg_incoming_buffer.append(torch.tensor(eeg_data).float())
    if ecg_data is not None:
        ecg_incoming_buffer.append(torch.tensor(ecg_data).float())

    # Schedule adaptation for EEG
    if len(eeg_incoming_buffer) >= 16:
        eeg_batch = torch.stack(eeg_incoming_buffer[:16])
        background_tasks.add_task(adapt_model, model, eeg_batch, eeg_adapt_buffer, eeg_gen_samples, eeg_gen_labels, device)
        del eeg_incoming_buffer[:16]

    # Schedule adaptation for ECG
    if len(ecg_incoming_buffer) >= 16:
        ecg_batch = torch.stack(ecg_incoming_buffer[:16])
        background_tasks.add_task(adapt_model, model_ECG, ecg_batch, ecg_adapt_buffer, ecg_gen_samples, ecg_gen_labels, device)
        del ecg_incoming_buffer[:16]
    # Return inference immediately
    result = infer(eeg_data, ecg_data)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
