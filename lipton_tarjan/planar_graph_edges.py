import os
import numpy as np
from numba import jitclass, numpy_support
from numba.types import int32
import datetime


@jitclass([('_vertex1', int32[:]),
        ('_vertex2', int32[:]),
        ('_index', int32[:]),
        ('_vertex1_next_edge_index', int32[:]),
        ('_vertex1_previous_edge_index', int32[:]),
        ('_vertex2_next_edge_index', int32[:]),
        ('_vertex2_previous_edge_index', int32[:]),
        ('_size', int32)])
class PlanarGraphEdges:

    def __init__(self, capacity):

        self._vertex1, self._vertex2, self._index, self._vertex1_next_edge_index, \
                self._vertex1_previous_edge_index, self._vertex2_next_edge_index, \
                self._vertex2_previous_edge_index = self._allocate_data(capacity)
        self._size = 0

    @property
    def size(self):

        return self._size

    @property
    def vertex1(self):

        return self._vertex1[:self._size]

    @property
    def vertex2(self):

        return self._vertex2[:self._size]

    @property
    def index(self):

        return self._index[:self._size]

    def append(self, vertex1, vertex2):

        self._vertex1[self._size] = vertex1
        self._vertex2[self._size] = vertex2

        self._size += 1

    def extend(self, edges):

        self._vertex1 = np.concatenate((self._vertex1[:self._size], edges._vertex1[:edges._size]))
        self._vertex2 = np.concatenate((self._vertex2[:self._size], edges._vertex2[:edges._size]))
        self._index = np.concatenate((self._index[:self._size], edges._index[:edges._size]))
        self._vertex1_next_edge_index = np.concatenate((self._vertex1_next_edge_index[:self._size],
                edges._vertex1_next_edge_index[:edges._size]))
        self._vertex1_previous_edge_index = \
                np.concatenate((self._vertex1_previous_edge_index[:self._size],
                edges._vertex1_previous_edge_index[:edges._size]))
        self._vertex2_next_edge_index = np.concatenate((self._vertex2_next_edge_index[:self._size],
                edges._vertex2_next_edge_index[:edges._size]))
        self._vertex2_previous_edge_index = \
                np.concatenate((self._vertex2_previous_edge_index[:self._size],
                edges._vertex2_previous_edge_index[:edges._size]))

        self._index[self._size:] += self._size

        self._size += edges.size

    def increase_capacity(self, capacity):

        vertex1, vertex2, index, vertex1_next_edge_index, vertex1_previous_edge_index, \
                vertex2_next_edge_index, vertex2_previous_edge_index = \
                self._allocate_data(capacity)

        vertex1[:self._size] = self._vertex1[:self._size]
        vertex2[:self._size] = self._vertex2[:self._size]
        vertex1_next_edge_index[:self._size] = self._vertex1_next_edge_index[:self._size]
        vertex1_previous_edge_index[:self._size] = self._vertex1_previous_edge_index[:self._size]
        vertex2_next_edge_index[:self._size] = self._vertex2_next_edge_index[:self._size]
        vertex2_previous_edge_index[:self._size] = self._vertex2_previous_edge_index[:self._size]

        self._vertex1 = vertex1
        self._vertex2 = vertex2
        self._index = index
        self._vertex1_next_edge_index = vertex1_next_edge_index
        self._vertex1_previous_edge_index = vertex1_previous_edge_index
        self._vertex2_next_edge_index = vertex2_next_edge_index
        self._vertex2_previous_edge_index = vertex2_previous_edge_index

    def _allocate_data(self, capacity):

        vertex1 = np.zeros(capacity, dtype=np.int32)
        vertex2 = np.zeros(capacity, dtype=np.int32)
        index = np.arange(capacity).astype(np.int32)
        vertex1_next_edge_index = np.zeros(capacity, dtype=np.int32)
        vertex1_previous_edge_index = np.zeros(capacity, dtype=np.int32)
        vertex2_next_edge_index = np.zeros(capacity, dtype=np.int32)
        vertex2_previous_edge_index = np.zeros(capacity, dtype=np.int32)

        return vertex1, vertex2, index, vertex1_next_edge_index, vertex1_previous_edge_index, \
                vertex2_next_edge_index, vertex2_previous_edge_index

    def get_opposite_vertex(self, edge_index, vertex):

        if vertex == self._vertex1[edge_index]:
            return self._vertex2[edge_index]

        return self._vertex1[edge_index]

    def set_next_edge(self, edge_index, vertex, other_edge_index):

        self._set_next_edge_only(edge_index, vertex, other_edge_index)
        self._set_previous_edge_only(other_edge_index, vertex, edge_index)

    def set_previous_edge(self, edge_index, vertex, other_edge_index):

        self._set_previous_edge_only(edge_index, vertex, other_edge_index)
        self._set_next_edge_only(other_edge_index, vertex, edge_index)

    def _set_next_edge_only(self, edge_index, vertex, other_edge_index):

        if vertex == self._vertex1[edge_index]:
            self._vertex1_next_edge_index[edge_index] = other_edge_index
        else:
            self._vertex2_next_edge_index[edge_index] = other_edge_index

    def _set_previous_edge_only(self, edge_index, vertex, other_edge_index):

        if vertex == self._vertex1[edge_index]:
            self._vertex1_previous_edge_index[edge_index] = other_edge_index
        else:
            self._vertex2_previous_edge_index[edge_index] = other_edge_index

    def get_next_edge_index(self, edge_index, vertex):

        if vertex == self._vertex1[edge_index]:
            return self._vertex1_next_edge_index[edge_index]

        return self._vertex2_next_edge_index[edge_index]

    def get_previous_edge_index(self, edge_index, vertex):

        if vertex == self._vertex1[edge_index]:
            return self._vertex1_previous_edge_index[edge_index]

        return self._vertex2_previous_edge_index[edge_index]

    def get_string_lines(self):

        lines = []

        for edge_index in range(self._size):

            lines.append(
                    '\t\tEdge(#{0}, v1={1}, v2={2}, ccw1={3}, cw1={4}, ccw2={5}, cw2={6})'.format(
                    self._index[edge_index],
                    self._vertex1[edge_index],
                    self._vertex2[edge_index],
                    self._vertex1_next_edge_index[edge_index],
                    self._vertex1_previous_edge_index[edge_index],
                    self._vertex2_next_edge_index[edge_index],
                    self._vertex2_previous_edge_index[edge_index]))

        return lines


planar_graph_edges_nb_type = PlanarGraphEdges.class_type.instance_type
