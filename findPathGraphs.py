# Estudo sobre decision trees utilizando o livro
# Machine Learning an algorithmic perspective

graph = {'A': ['B', 'C'], 'B': ['C', 'D'], 'C': ['D'], 'D': ['C'], 'E': ['F'], 'F': ['C']
}
print(graph)

def findPath(graph, start, end, pathSoFar):
      pathSoFar = pathSoFar + [start]

      print(f"Checking {start} -> {end} with current path: {pathSoFar}")

      if start == end:
            return pathSoFar
      if start not in graph:
            for node in graph[start]:
                  if node not in pathSoFar:
                        newpath = findPath(graph, node, end, pathSoFar)
                        return newpath
      
      return None

print(findPath(graph, 'B', 'D', []))