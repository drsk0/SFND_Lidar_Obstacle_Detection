/* \author Aaron Brown */
// Quiz on implementing kd tree
#ifndef KDTREE_H_
#define KDTREE_H_

#include "render/render.h"

// Structure to represent node of kd tree
struct Node {
  std::vector<float> point;
  int id;
  Node *left;
  Node *right;

  Node(std::vector<float> arr, int setId)
      : point(arr), id(setId), left(NULL), right(NULL) {}

  ~Node() {
    delete left;
    delete right;
  }
};

struct KdTree {
  Node *root;

  KdTree() : root(NULL) {}

  ~KdTree() { delete root; }

  void insert(std::vector<float> &point, int id) {
    if (root == NULL) {
      root = new Node(point, id);
      return;
    }
    insertHelper(root, 0, point, id);
  }

  void insertHelper(Node *root, const int level,
                    const std::vector<float> &point, const int id) {
    if (root == NULL)
      return;
    int ix = level % 3;

    if (point[ix] < root->point[ix]) {
      if (root->left == NULL) {
        root->left = new Node(point, id);
        return;
      } else {
        insertHelper(root->left, level + 1, point, id);
      }
    } else { // point[ix] >= root->point[ix]
      if (root->right == NULL) {
        root->right = new Node(point, id);
        return;
      } else {
        insertHelper(root->right, level + 1, point, id);
      }
    }

    return;
  }

  // return a list of point ids in the tree that are within distance of target
  std::vector<int> search(const std::vector<float> &target,
                          const float distanceTol) const {
    std::vector<int> acc;
    searchHelper(acc, root, 0, target, distanceTol);
    return acc;
  }

  void searchHelper(std::vector<int> &acc, const Node *root, const int level,
                    const std::vector<float> &target,
                    const float distanceTol) const {
    if (root == NULL) {
      return;
    }

    int ix = level % 3;
    if (abs(root->point[0] - target[0]) <= distanceTol &&
        abs(root->point[1] - target[1]) <= distanceTol &&
        abs(root->point[2] - target[2]) <= distanceTol) {
      if (pow(root->point[0] - target[0], 2) +
              pow(root->point[1] - target[1], 2) +
              pow(root->point[2] - target[2], 2) <=
          pow(distanceTol, 2)) {
        acc.push_back(root->id);
      }
      searchHelper(acc, root->left, level + 1, target, distanceTol);
      searchHelper(acc, root->right, level + 1, target, distanceTol);
    } else // root not within distance
    {
      if (target[ix] - distanceTol < root->point[ix]) {
        searchHelper(acc, root->left, level + 1, target, distanceTol);
      }
      if (target[ix] + distanceTol > root->point[ix]) {
        searchHelper(acc, root->right, level + 1, target, distanceTol);
      }
    }
  }
};

#endif // KDTREE_H_
