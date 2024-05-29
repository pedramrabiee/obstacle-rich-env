import torch


def vstack_to_tensor(tensor: torch.Tensor, to_append: torch.Tensor) -> torch.Tensor:
    if to_append.numel() == 0:
        return tensor
    if tensor is None:
        return to_append
    else:
        return torch.cat((tensor, to_append), dim=0)


class FixedSizeTensorStack:
    """
        A fixed-size stack of tensors.

        This class maintains a stack of tensors with a fixed maximum length for each batch.
        When new elements are appended, the oldest elements are removed to maintain the fixed size.
        It supports appending new elements, popping the most recent elements, and automatically
        filling vacated positions with an initial value.

        Attributes:
        - max_length (int): The maximum number of elements the stack can hold.
        - batch_size (int): The number of batches.
        - init_value (float): The initial value to fill the stack with.
        - stack (torch.Tensor): The tensor representing the stack.
        """

    def __init__(self, batch_size, max_length, init_value=0.0):
        """
            Initializes a fixed-size stack of tensors.

            Parameters:
            - batch_size (int): The number of batches.
            - max_length (int): The maximum number of elements in the stack.
            - init_value (float): The initial value to fill the stack with.
        """

        self.max_length = max_length
        self.batch_size = batch_size
        self.init_value = init_value
        self.stack = torch.ones((batch_size, max_length), dtype=torch.float64) * init_value

    def append(self, x):
        """
            Appends a new tensor to the stack. Removes the oldest element.

            Parameters:
            - x (torch.Tensor): A tensor to append. Should have shape (batch_size,) or (batch_size, 1).
        """
        if x.ndim == 1:
            x.unsqueeze_(1)
        assert x.shape[0] == self.batch_size

        self.stack = self.stack[:, 1:]  # remove the oldest element
        self.stack = torch.hstack((self.stack, x))

    def pop(self):
        """
           Removes and returns the most recent element from the stack.
           The vacated position is filled from left with the initial value.

           Returns:
           - torch.Tensor: The most recent element.

       """
        x = self.stack[:, -1].unsqueeze(-1)
        self.stack = self.stack[:, :-1]
        self._appendleft(torch.ones((self.batch_size, 1), dtype=torch.float64) * self.init_value)
        return x

    def reset(self, rows_to_reset=None):
        """
           Resets the stack or specific rows in the stack to the initial value.

           Parameters:
           - rows_to_reset (list of int, optional): List of row indices to reset. If None, resets the entire stack.
       """
        if rows_to_reset is None:
            self.__init__(self.batch_size, self.max_length, self.init_value)
        else:
            overwrite_row = torch.ones((1, self.max_length), dtype=torch.float64) * self.init_value
            for row in rows_to_reset:
                self.stack[row] = overwrite_row

    def _appendleft(self, x):
        """
           Appends a new tensor to the beginning of the stack.

           Parameters:
           - x (torch.Tensor): A tensor to append to the left. Should have shape (batch_size,) or (batch_size, 1).
       """
        if x.ndim == 1:
            x.unsqueeze_(1)
        assert x.shape[0] == self.batch_size

        self.stack = torch.hstack((x, self.stack))

    def __len__(self):
        return self.stack.shape[1]

    def __repr__(self):
        return self.stack.__repr__()

    @property
    def shape(self):
        return self.stack.shape

# # Uncomment for Example Usage
#
# # Create a FixedSizeTensorStack with batch_size=3, max_length=5, and init_value=0.5
# dq = FixedSizeTensorStack(batch_size=3, max_length=5, init_value=0.5)
#
# # Print the initial state of the stack
# print("Initial stack:")
# print(dq)
#
# # Append tensors to the stack
# dq.append(torch.tensor([1.0, 2.0, 3.0]))
# dq.append(torch.tensor([2.0, 1.0, 1.0]))
# dq.append(torch.tensor([-1.0, -2.0, -3.0]))
# dq.append(torch.tensor([-2.0, -1.0, -1.0]))
# dq.append(torch.tensor([0.0, 0.0, 0.0]))
# dq.append(torch.tensor([0.0, 0.0, 0.0]))
# dq.append(torch.tensor([0.0, 0.0, 0.0]))
#
# # Print the stack after appending
# print("\nStack after appending:")
# print(dq)
#
# # Reset specific rows in the stack
# dq.reset(rows_to_reset=[0, 2])
#
# # Print the stack after resetting specific rows
# print("\nStack after resetting rows 0 and 2:")
# print(dq)
#
# # Pop elements from the stack
# dq.pop()
# dq.pop()
#
# # Print the stack after popping elements
# print("\nStack after popping elements:")
# print(dq)
