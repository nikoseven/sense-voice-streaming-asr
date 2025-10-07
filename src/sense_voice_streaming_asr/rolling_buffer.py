import numpy as np


class RollingBuffer:
    def __init__(self, shape, dtype="float32", capacity=1000):
        """
        :param shape: Shape of each element in the buffer
        :param dtype: Data type of elements
        :param capacity: Maximum capacity of the buffer
        """
        self.capacity = capacity
        self.buffer = np.zeros((capacity, *shape), dtype=dtype)
        self.count = 0  # Current number of elements in buffer
        self.p = 0  # Logical position pointing to the oldest element

    def __len__(self):
        return self.count

    @property
    def head(self) -> int:
        return self.p

    @property
    def tail(self) -> int:
        return self.p + self.count

    @property
    def is_full(self) -> bool:
        return self.count == self.capacity

    @property
    def space_left(self) -> int:
        return self.capacity - self.count

    def pop_head(self) -> np.ndarray:
        """
        Pop and return the head element (the oldest one).
        """
        if self.count == 0:
            raise IndexError("Cannot pop from an empty buffer")

        head_elem = self.buffer[self.p % self.capacity]
        self.p += 1
        self.count -= 1
        return head_elem

    def drop_buffer_before(self, min_keep: int) -> None:
        """
        Discard all elements that are positioned before ``min_keep`` in the logical
        buffer index space.

        Parameters
        ----------
        min_keep : int
            The logical index (relative to the start of the buffer) that
            should become the new head of the buffer.  Elements with a
            logical index < ``min_keep`` are removed.

        Notes
        -----
        * ``self.p`` is the logical index of the current head (oldest
          element).  ``self.p + self.count`` is the logical index of
          the logical tail (the index after the newest element).
        * ``min_keep`` is clamped to the range ``[self.p, self.p + self.count]``.
        * The method updates ``self.p`` and ``self.count`` in‑place.
        """
        if not isinstance(min_keep, (int, np.integer)):
            raise TypeError("min_keep must be an integer")

        # Clamp the requested new head to the valid range.
        #   - It cannot be before the current head (no backward move).
        #   - It cannot be beyond the logical tail.
        new_head = max(self.p, min(min_keep, self.p + self.count))

        # Reduce the count and update the head pointer.
        self.count -= new_head - self.p
        self.p = new_head

    def append(self, x):
        if self.count == self.capacity:
            self.p += 1  # overwrite oldest
        else:
            self.count += 1
        self.buffer[(self.p + self.count - 1) % self.capacity] = x

    def extend(self, xs: np.ndarray):
        n = xs.shape[0]
        if n == 0:
            return

        if xs.shape[1:] != self.buffer.shape[1:]:
            raise ValueError(
                f"Data shape {xs.shape[1:]} does not match buffer element shape {self.buffer.shape[1:]}"
            )

        if n > self.capacity:
            xs = xs[-self.capacity :]
            n = self.capacity
            self.p = self.tail
        else:
            if self.count + n > self.capacity:
                self.p += (self.count + n) - self.capacity

        self.count = min(self.count + n, self.capacity)

        tail = self.tail

        physical_start = (tail - n) % self.capacity

        if physical_start + n <= self.capacity:
            self.buffer[physical_start : physical_start + n] = xs
        else:
            part1_size = self.capacity - physical_start
            self.buffer[physical_start:] = xs[:part1_size]

            part2_size = n - part1_size
            self.buffer[:part2_size] = xs[part1_size:]

    def get_all(self) -> np.ndarray:
        if self.count == 0:
            return np.empty((0, *self.buffer.shape[1:]), dtype=self.buffer.dtype)

        physical_p = self.p % self.capacity
        if physical_p + self.count <= self.capacity:
            return self.buffer[physical_p : physical_p + self.count]
        else:
            part1 = self.buffer[physical_p:]
            part2_len = (physical_p + self.count) % self.capacity
            part2 = self.buffer[:part2_len]
            return np.concatenate((part1, part2))

    def __getitem__(self, key):
        """
        Efficiently access elements by index or slice (using absolute logical indices).

        - Supports integer indexing (including negative, counting from the end).
        - Supports slicing (step=1 only), returns view if data is continuous, otherwise returns copy.
        """
        if isinstance(key, slice):
            if key.step is not None and key.step != 1:
                raise NotImplementedError("Slicing with step != 1 is not supported")

            slice_start, slice_stop = key.start, key.stop

            # 1. Resolve None and negative indices to absolute logical indices
            if slice_start is None:
                start_abs = self.p
            elif slice_start < 0:
                start_abs = self.tail + slice_start
            else:
                start_abs = slice_start

            if slice_stop is None:
                stop_abs = self.tail
            elif slice_stop < 0:
                stop_abs = self.tail + slice_stop
            else:
                stop_abs = slice_stop

            # 2. Clamp the absolute slice to the valid range of the buffer's content
            start_abs = max(self.p, min(self.tail, start_abs))
            stop_abs = max(self.p, min(self.tail, stop_abs))

            # 3. Calculate slice length and relative start offset
            slice_len = stop_abs - start_abs
            if slice_len <= 0:
                return np.empty((0, *self.buffer.shape[1:]), dtype=self.buffer.dtype)

            start_offset = start_abs - self.p

            # 4. Calculate physical start in the numpy array
            physical_start = (self.p + start_offset) % self.capacity

            # 5. Copy data, handling wrap-around
            if physical_start + slice_len <= self.capacity:
                return self.buffer[physical_start : physical_start + slice_len]
            else:
                part1 = self.buffer[physical_start:]
                part2_len = (physical_start + slice_len) % self.capacity
                part2 = self.buffer[:part2_len]
                return np.concatenate((part1, part2))

        if isinstance(key, (int, np.integer)):
            # Handle negative index, relative to tail
            if key < 0:
                key += self.tail

            # Check if the absolute index is in the valid range
            if not (self.p <= key < self.tail):
                raise IndexError("Index out of range")

            physical_index = key % self.capacity
            return self.buffer[physical_index]

        raise TypeError(
            f"Buffer indices must be integers or slices, not {type(key).__name__}"
        )
