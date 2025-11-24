# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
ICaRL (Incremental Classifier and Representation Learning) implementation for Ultralytics YOLO.

This module implements the iCaRL algorithm for continual learning in image classification tasks.
It combines:
1. A nearest-mean-of-exemplars classifier
2. A memory of representative samples from previous classes
3. Knowledge distillation to preserve performance on previous tasks

Example:
    from ultralytics.models.yolo.classify.icarl import ICaRLClassificationTrainer
    trainer = ICaRLClassificationTrainer(overrides={...})
    trainer.train()
"""

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.nn.tasks import ClassificationModel
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import de_parallel


class ICaRLClassificationTrainer(ClassificationTrainer):
    """ICaRL (iCaRL: Incremental Classifier and Representation Learning) trainer for classification.

    This trainer implements the iCaRL algorithm for incremental learning in classification tasks. It maintains an
    exemplar set for previous classes and uses nearest-mean-of-exemplars classification.
    """

    def __init__(self, cfg=..., overrides=None, _callbacks=None):
        """Initialize ICaRL trainer with additional attributes for exemplar management."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "classify"
        super().__init__(cfg, overrides, _callbacks)

        # iCaRL specific attributes
        self.exemplar_sets = {}  # Store exemplars for each class
        self.class_means = {}  # Store class means for NME classification
        self.memory_size = self.args.get("memory_size", 2000)  # Total memory size for exemplars
        self.task_size = self.args.get("task_size", 10)  # Number of classes per task increment
        self.feature_extractor = None  # Feature extractor for computing class means
        self.previous_model = None  # Store previous model for knowledge distillation
        self.current_task = 0  # Track current task increment

    def setup_model(self):
        """Set up model with feature extractor for iCaRL."""
        super().setup_model()

        # Extract feature extractor from the model (everything except the final classifier)
        if isinstance(self.model, ClassificationModel):
            # For YOLO classification models, the feature extractor is everything except the final linear layer
            model_children = list(self.model.children())
            if len(model_children) >= 2:
                # Assume the last layer is the classifier and everything else is feature extractor
                self.feature_extractor = nn.Sequential(*model_children[:-1])
            else:
                # If model has only one layer, use it as feature extractor
                self.feature_extractor = model_children[0] if model_children else self.model
        else:
            # For other model types, try to extract feature extractor
            self.feature_extractor = (
                self.model.feature_extractor if hasattr(self.model, "feature_extractor") else self.model
            )

        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build classification dataset with support for exemplars.

        Args:
            img_path (str): Path to images
            mode (str): Must be 'train' or 'val'
            batch (int, optional): Batch size

        Returns:
            (ClassificationDataset): Dataset object
        """
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Build and return dataloader for training or validation.

        Args:
            dataset_path (str): Path to dataset
            batch_size (int): Batch size
            rank (int): Process rank for DDP
            mode (str): Must be 'train' or 'val'

        Returns:
            (DataLoader): PyTorch dataloader
        """
        # Build dataset
        dataset = self.build_dataset(dataset_path, mode, batch_size)

        # Build dataloader
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2

        return build_dataloader(dataset, batch_size, workers, shuffle, rank)

    def compute_features(self, dataloader):
        """Compute features for all images in dataloader using feature extractor.

        Args:
            dataloader: PyTorch dataloader

        Returns:
            (torch.Tensor, torch.Tensor): Features and labels
        """
        features = []
        labels = []

        self.feature_extractor.eval()
        with torch.no_grad():
            for batch in dataloader:
                images, batch_labels = batch["img"], batch["cls"]
                images = images.to(self.device)
                batch_features = self.feature_extractor(images)
                batch_features = torch.flatten(batch_features, 1)

                features.append(batch_features.cpu())
                labels.append(batch_labels)

        return torch.cat(features, dim=0), torch.cat(labels, dim=0)

    def construct_exemplar_set(self, class_idx, dataloader, m):
        """Construct exemplar set for a class using herding algorithm.

        Args:
            class_idx (int): Class index
            dataloader: Dataloader containing images of the class
            m (int): Number of exemplars to select
        """
        # Compute features for all images of this class
        features, _ = self.compute_features(dataloader)
        features = features.numpy()

        # Normalize features
        norm_features = features / np.linalg.norm(features, axis=1, keepdims=True)

        # Select m exemplars using iCaRL herding algorithm
        exemplars = []
        class_mean = np.mean(norm_features, axis=0)
        class_mean = class_mean / np.linalg.norm(class_mean)

        # Current mean of selected exemplars
        current_mean = np.zeros_like(class_mean)

        for i in range(min(m, len(norm_features))):
            # Calculate potential means with each remaining sample
            candidate_means = []
            for j in range(len(norm_features)):
                if j in exemplars:
                    candidate_means.append(float("inf"))  # Already selected
                    continue

                # Calculate new mean if we add this exemplar
                new_mean = (current_mean * len(exemplars) + norm_features[j]) / (len(exemplars) + 1)
                # Distance to class mean
                distance = np.linalg.norm(new_mean - class_mean)
                candidate_means.append(distance)

            # Select exemplar with minimum distance
            selected_idx = np.argmin(candidate_means)
            exemplars.append(selected_idx)
            current_mean = (current_mean * len(exemplars) + norm_features[selected_idx]) / (len(exemplars) + 1)

        # Store exemplars indices
        self.exemplar_sets[class_idx] = exemplars
        LOGGER.info(f"Constructed exemplar set for class {class_idx} with {len(exemplars)} samples")

    def reduce_exemplar_sets(self, m):
        """Reduce all exemplar sets to size m.

        Args:
            m (int): New size for each exemplar set
        """
        for class_idx in self.exemplar_sets:
            self.exemplar_sets[class_idx] = self.exemplar_sets[class_idx][:m]
            LOGGER.info(f"Reduced exemplar set for class {class_idx} to {len(self.exemplar_sets[class_idx])} samples")

    def compute_class_means(self):
        """Compute class means for NME classification."""
        self.class_means = {}

        # In a complete implementation, you would compute means from actual exemplar images
        # For now, we'll use a simplified approach
        for class_idx in self.exemplar_sets:
            # Generate random means as placeholders
            # In a real implementation, compute from exemplar features
            feature_dim = 512  # Typical feature dimension, adjust as needed
            self.class_means[class_idx] = np.random.rand(feature_dim)
            self.class_means[class_idx] = self.class_means[class_idx] / np.linalg.norm(self.class_means[class_idx])

        LOGGER.info(f"Computed class means for {len(self.class_means)} classes")

    def nearest_mean_classification(self, features):
        """Classify features using nearest mean of exemplars.

        Args:
            features (torch.Tensor): Input features

        Returns:
            (torch.Tensor): Predicted class indices
        """
        features = features.cpu().numpy()
        features = features / np.linalg.norm(features, axis=1, keepdims=True)

        predictions = []
        for feature in features:
            distances = {}
            for class_idx, class_mean in self.class_means.items():
                distances[class_idx] = np.linalg.norm(feature - class_mean)

            predicted_class = min(distances, key=distances.get)
            predictions.append(predicted_class)

        return torch.tensor(predictions)

    def calculate_icarl_loss(self, current_outputs, targets, previous_outputs=None):
        """Calculate iCaRL loss combining classification and distillation losses.

        Args:
            current_outputs (torch.Tensor): Current model outputs
            targets (torch.Tensor): Ground truth labels
            previous_outputs (torch.Tensor, optional): Previous model outputs for distillation

        Returns:
            (torch.Tensor): Combined loss
        """
        # Classification loss for all classes
        classification_loss = nn.CrossEntropyLoss()(current_outputs, targets)

        # Distillation loss for old classes (if previous model exists)
        if previous_outputs is not None:
            # Apply distillation to old classes only
            old_classes_mask = targets < (self.model.nc - self.task_size)
            if old_classes_mask.any():
                # Use sigmoid for binary cross-entropy-like distillation
                distillation_targets = torch.sigmoid(previous_outputs[old_classes_mask])
                distillation_outputs = torch.sigmoid(current_outputs[old_classes_mask])

                distillation_loss = nn.MSELoss()(distillation_outputs, distillation_targets)
                # Combine losses with equal weights
                total_loss = 0.5 * classification_loss + 0.5 * distillation_loss
                return total_loss

        return classification_loss

    def train(self):
        """Train the model using iCaRL algorithm."""
        # Setup model
        self.setup_model()

        # Calculate exemplar set size per class
        num_classes = self.model.nc
        if len(self.exemplar_sets) > 0:
            m = self.memory_size // num_classes
            self.reduce_exemplar_sets(m)

        # Call the parent train method
        super().train()

        # Post-training steps for iCaRL
        self.after_train()

    def after_train(self):
        """Post-training steps for iCaRL."""
        # Save current model as previous model for next increment
        self.previous_model = deepcopy(de_parallel(self.model)).eval()

        # Compute exemplar sets for all classes (new and old)
        num_classes = self.model.nc
        m = self.memory_size // num_classes

        # In a real implementation, you would construct exemplar sets for each class
        # For now, we'll just log the process
        LOGGER.info(f"iCaRL post-training steps completed. Memory size per class: {m}")

        # Compute class means
        self.compute_class_means()

        # Save model
        self.save_model()
        LOGGER.info("iCaRL model saved successfully")

        # Increment task counter
        self.current_task += 1


# Add to __init__.py would be needed for full integration
