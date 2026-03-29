import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Use the y_true and y_pred from your evaluate.py logic
# This is a placeholder for your report
y_true = [0, 1, 2] # Example
y_pred = [0, 1, 2] # Example

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Sentiment Analysis Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("Heatmap saved as confusion_matrix.png")