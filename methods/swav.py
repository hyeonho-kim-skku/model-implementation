import torch
import torch.nn as nn
import torch.nn.functional as F

class SwAV(nn.Module):
    def __init__(self, model, dim=128, hidden_dim=2048, n_prototypes=3000,
                 epsilon=0.05, temperature=0.1, sinkhorn_iterations=3, queue_length=4096):
        """
        model: base encoder
        dim: output dimension of projection head
        hidden_dim: hidden dimension of projection head
        n_prototypes: number of prototypes (clusters)
        epsilon: regularization parameter for Sinkhorn-Knopp algorithm
        temperature: temperature parameter for softmax
        sinkhorn_iterations: number of iterations for Sinkhorn-Knopp
        queue_length: number of samples to keep in queue (for small batch training)
        """
        super().__init__()
        self.epsilon = epsilon
        self.temperature = temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.queue_length = queue_length

        # 1. Encoder
        self.encoder =model
        feat_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        # 2. Projection Head
        self.projection_head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

        # 3. Prototypes
        self.prototypes = nn.Linear(dim, n_prototypes, bias=False)

        # 4. Queue (Optional) for storing previous features
        # use when training with small batch sizes
        if queue_length > 0:
            self.register_buffer("queue", torch.zeros(queue_length, dim))
        else:
            self.queue = None
        
        self.model = self.encoder

    def forward(self, batch):
        """
        batch: (crops, labels)
        """
        crops, _ = batch
        bs = crops[0].size(0) # batch size

        # 1. Separate global and local crops
        global_crops = crops[:2]
        local_crops = crops[2:]

        # 2. Forward pass
        # a. Global crops
        global_inputs = torch.cat(global_crops, dim=0) # (2*B, 3, 32, 32)
        global_features = self.projection_head(self.encoder(global_inputs)) # (2*B, dim)

        # b. Local crops
        local_inputs = torch.cat(local_crops, dim=0) # (V*B, 3, 16, 16)
        local_features = self.projection_head(self.encoder(local_inputs)) # (V*B, dim)

        # 3. Concatenate features
        features = torch.cat([global_features, local_features], dim=0) # ((2+V)*B, dim)
        
        # 4. Normalize features
        features = F.normalize(features, dim=1) # ((2+V)*B, dim)

        # 5. Normalize prototypes and update
        with torch.no_grad():
            w = self.prototypes.weight.data.clone() # (n_prototypes, dim)
            w = F.normalize(w, dim=1) # normalize each prototype
            self.prototypes.weight.copy_(w)

        # 6. Compute scores (between prototypes and features)
        # linear layer: input @ weight.T (+ bias, now None)
        # features: ((2+V)*B, dim), prototypes.weight: (n_prototypes, dim)
        scores = self.prototypes(features) # ((2+V)*B, n_prototypes)

        # Use queue when queue is full
        use_queue = False
        if self.queue is not None and (not torch.all(self.queue[-1,:] == 0)):
            use_queue = True

        # 7. Compute assignments for each global crop
        q_list = [] # list of assignments for each global crop
        for i in range(len(global_crops)):
            # only for global crops (first 2*B samples)
            with torch.no_grad():
                global_scores = scores[i*bs:(i+1)*bs] # (2*B, n_prototypes)

                if use_queue:
                    queue_scores = self.prototypes(self.queue) # (queue_length, n_prototypes)
                    sinkhorn_input = torch.cat([global_scores, queue_scores], dim=0) # (B + queue_length, n_prototypes)
                else:
                    sinkhorn_input = global_scores
                
                q = self.sinkhorn(sinkhorn_input)

                if use_queue:
                    q = q[:bs]

                q_list.append(q) # append assignments
        
        q = torch.cat(q_list, dim=0) # (2*B, n_prototypes)
        
        # 8. Compute loss
        # Cross entropy between assignments (q) and probabilities (p)
        # p: softmax(scores / temperature)
        log_probs = F.log_softmax(scores / self.temperature, dim=1) # ((2+V)*B, n_prototypes)

        total_loss = 0
        n_loss_terms = 0

        for i in range(len(global_crops)): # for each global crop
            for j in range(len(crops)): # for each crop (global + local)
                if i == j: # skip same view
                    continue

                current_q = q[i*bs : (i+1)*bs] # (B, n_prototypes)
                current_log_probs = log_probs[j*bs : (j+1)*bs] # (B, n_prototypes)

                # Cross-entropy loss
                loss = - torch.mean(torch.sum(current_q * current_log_probs, dim=1))
                total_loss += loss
                n_loss_terms += 1

        # 9. Update queue
        if self.queue_length is not None:
            with torch.no_grad():
                # global features from the current batch
                self._dequeue_and_enqueue(features[:2*bs].detach())

        return total_loss / n_loss_terms
    
    @ torch.no_grad()
    def sinkhorn(self, scores):
        Q = torch.exp(scores / self.epsilon).t()  # (K, B)
        B = Q.shape[1] # number of samples to assign
        K = Q.shape[0] # how many prototypes

        # make the matrix sums to 
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for _ in range(self.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
            Q /= sum_of_cols
            Q /= B

        Q *= B # the colomns must sum to 1 so that Q is an assignment
        return Q.t()
        
    @ torch.no_grad()
    def _dequeue_and_enqueue(self, global_features):
        features_size = global_features.size(0)

        # use queue when features for assignments are less than queue length
        if features_size < self.queue_length:
            # replace the oldest features in the queue with the new ones (FIFO)
            self.queue[features_size:] = self.queue[:-features_size].clone()
            self.queue[:features_size] = global_features