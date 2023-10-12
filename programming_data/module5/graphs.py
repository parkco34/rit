#!/usr/bin/env python

class Node(object):
    def __init__(self, name):
        """ Assumes name is a string """
        self._name = name

    def get_name(self):
        return self._name

    def __str__(self):
        return self._name


class Edge(object):
    def __init__(self, src, dest):
        """ Assumes src and dest are nodes """
        self._src = src # Parent node
        self._dest = dest # Child node

    def get_source(self):
        return self._src

    def get_dest(self):
        return self._dest

    def __str__(self):
        return self._src.get_name() + '->' + self._dest.get_name()


class Weighted_edge(Edge):
    def __init__(self, src, dest, weight=1.0):
        """ Assumes src and dest are nodes, weight a number """
        self._src = src
        self._dest = dest
        self._weight = weight

    def get_weight(self):
        return self._weight

    def __str__(self):
        return  (f"{self._src.get_name()} -> ({self._weight})" +
                 F"{self._dest.get_name()}")


class Diagraph(object):
    # Nodes is a list of the nodes in the graph
    # Edges is a dict mapping each node to a list of its children
    def __init__(self):
        self._nodes = []
        self._edges = {}

    def add_node(self, node):
        if node in self._nodes:
            raise ValueError("Duplicate Node")

        else:
            self._nodes.append(node)
            self._edges[node] = []

    def add_edge(self, edge):
        src = edge.get_source()
        dest = edge.get_destination()

        if not (src in self._nodes and dest in self._nodes):
            raise ValueError("Node not in graph")
        self._edges[src].append(dest)

    def children_of(self, node):
        return self._edges[node]

    def has_node(self, node):
        return node in self._nodes

    def __str__(self):
        result = ""

        for src in self._nodes:
            for dest in self._edges[src]:
                result = (result + src.get_name() + "->" + dest.get_name() + "\n")

        return result[:-1] # omit final newline

class Graph(Digraph):
    # Inherits all methods from Digraph except add_edge, which it overrides
    def add_edge(self, edge):
        Digraph.add_edge(self, edge)
        rev = Edge(edge.get_destination(), edge.get_source())
        Digraph.add_edge(self, rev)



