import typing as t

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger


class LogisticRegression:
    def __init__(self, learning_rate: float = 1e-4, num_iterations: int = 100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        """
        Implement your fitting function here.
        The weights and intercept should be kept in self.weights and self.intercept.
        """
        n_samples, n_features = inputs.shape

        self.weights = np.zeros(n_features)
        self.intercept = 0

        for i in range(self.num_iterations):
            linear_model = np.dot(inputs, self.weights) + self.intercept
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(inputs.T, (y_predicted - targets))
            db = (1 / n_samples) * np.sum(y_predicted - targets)

            self.weights -= self.learning_rate * dw
            self.intercept -= self.learning_rate * db

    def predict(self, inputs: npt.NDArray[float]) -> t.Tuple[t.Sequence[np.float64], t.Sequence[int]]:
        """
        Implement your prediction function here.
        The return should contains
        1. sample probabilty of being class_1
        2. sample predicted class
        """
        linear_model = np.dot(inputs, self.weights) + self.intercept
        y_predicted_probs = self.sigmoid(linear_model)
        y_predicted_classes = (y_predicted_probs >= 0.5).astype(int)

        return y_predicted_probs, y_predicted_classes

    def sigmoid(self, x):
        """
        Implement the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))


class FLD:
    """Implement FLD
    You can add arguments as you need,
    but don't modify those already exist variables.
    """
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        X0 = inputs[targets == 0]
        X1 = inputs[targets == 1]

        self.m0 = np.mean(X0, axis=0)
        self.m1 = np.mean(X1, axis=0)

        # sw, sb, fld
        S0 = np.dot((X0 - self.m0).T, (X0 - self.m0))
        S1 = np.dot((X1 - self.m1).T, (X1 - self.m1))
        self.sw = S0 + S1

        mean_diff = (self.m1 - self.m0).reshape(-1, 1)
        self.sb = np.dot(mean_diff, mean_diff.T)

        self.w = np.dot(np.linalg.inv(self.sw), (self.m1 - self.m0))

        # slope after project
        self.slope = self.w[1] / self.w[0]

    def predict(self, inputs: npt.NDArray[float]) -> t.Sequence[t.Union[int, bool]]:
        projections = np.dot(inputs, self.w)

        m0_proj = np.dot(self.m0, self.w)
        m1_proj = np.dot(self.m1, self.w)

        # choose near one
        dist_to_m0 = np.abs(projections - m0_proj)
        dist_to_m1 = np.abs(projections - m1_proj)
        predictions = (dist_to_m1 < dist_to_m0).astype(int)

        return predictions

    def plot_projection(self, inputs: npt.NDArray[float], targets: t.Sequence[int],
                        predictions: t.Sequence[int] = None):
        import matplotlib.pyplot as plt

        if predictions is None:  # for test_main.py's argument(no predictions)
            predictions = self.predict(inputs)

        plt.figure(figsize=(10, 8))

        # draw projection line
        x_min, x_max = inputs[:, 0].min() - 1, inputs[:, 0].max() + 1
        x_line = np.array([x_min, x_max])
        global_mean = (self.m0 + self.m1) / 2
        intercept = global_mean[1] - self.slope * global_mean[0]
        y_line = self.slope * x_line + intercept
        plt.plot(x_line, y_line, 'gray', linewidth=2, label='Projection line')

        # project all data to 1D space
        w_unit = self.w / np.linalg.norm(self.w)
        projected_points = np.zeros_like(inputs)
        for i in range(len(inputs)):
            v = inputs[i] - global_mean
            proj_length = np.dot(v, w_unit)
            projected_points[i] = global_mean + proj_length * w_unit

        # draw decision boundary
        m0_proj = np.dot(self.m0, self.w)
        m1_proj = np.dot(self.m1, self.w)
        decision_point = (m0_proj + m1_proj) / 2

        # decision boundary is perpendicular to w
        w_perpendicular = np.array([-self.w[1], self.w[0]])
        w_perpendicular = w_perpendicular / np.linalg.norm(w_perpendicular)

        # points on decision boundary
        decision_center = decision_point * self.w / np.dot(self.w, self.w)
        scale = np.array([x_min - x_max, x_max - x_min])
        decision_line_points = (decision_center[:, np.newaxis] + w_perpendicular[:, np.newaxis] * scale)

        plt.plot(decision_line_points[0, :], decision_line_points[1, :], 'b-', linewidth=2, label='Decision boundary')

        # draw data
        for i in range(len(inputs)):
            x, y = inputs[i]
            true_label = targets[i]
            pred_label = predictions[i]

            color = 'green' if true_label == pred_label else 'red'
            marker = 'o' if true_label == 0 else '^'
            plt.scatter(x, y, c=color, marker=marker, s=100, edgecolors='black')

            # draw project_to_1D dotted line
            plt.plot([x, projected_points[i, 0]], [y, projected_points[i, 1]], 'k--',
                     alpha=0.3, linewidth=0.8, zorder=1)

        # draw projection point on the projection line
        plt.scatter(projected_points[:, 0], projected_points[:, 1],
                    c='black', s=30, alpha=0.6, marker='o', zorder=2)

        plt.xlabel('Feature 27')
        plt.ylabel('Feature 30')
        plt.title(f'FLD Projection (slope={self.slope:.3f}, intercept={intercept:.3f})')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.savefig('fld_projection.png', dpi=300, bbox_inches='tight')
        plt.show()


def compute_auc(y_trues, y_preds):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_trues, y_preds)


def accuracy_score(y_trues, y_preds):
    return np.mean(y_trues == y_preds)


def main():
    # Read data
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # Part1: Logistic Regression
    x_train = train_df.drop(['target'], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df['target'].to_numpy()  # (n_samples, )
    print(y_train.shape)

    x_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    LR = LogisticRegression(
        learning_rate=5e-2,  # You can modify the parameters as you want
        num_iterations=1000,  # You can modify the parameters as you want
    )
    LR.fit(x_train, y_train)
    y_pred_probs, y_pred_classes = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)
    auc_score = compute_auc(y_test, y_pred_probs)
    logger.info(f'LR: Weights: {LR.weights[:5]}, Intercep: {LR.intercept}')
    logger.info(f'LR: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}')

    # Part2: FLD
    cols = ['27', '30']  # Dont modify
    x_train = train_df[cols].to_numpy()
    y_train = train_df['target'].to_numpy()
    x_test = test_df[cols].to_numpy()
    y_test = test_df['target'].to_numpy()

    FLD_ = FLD()
    """
    (TODO): Implement your code to
    1) Fit the FLD model
    2) Make prediction
    3) Compute the evaluation metrics

    Please also take care of the variables you used.
    """
    FLD_.fit(x_train, y_train)
    y_pred_fld = FLD_.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_fld)

    logger.info(f'FLD: m0={FLD_.m0}, m1={FLD_.m1} of {cols=}')
    logger.info(f'FLD: \nSw=\n{FLD_.sw}')
    logger.info(f'FLD: \nSb=\n{FLD_.sb}')
    logger.info(f'FLD: \nw=\n{FLD_.w}')
    logger.info(f'FLD: Accuracy={accuracy:.4f}')

    """
    (TODO): Implement your code below to plot the projection
    """
    FLD_.plot_projection(x_test, y_test, y_pred_fld)


if __name__ == '__main__':
    main()
