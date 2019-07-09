/*
 * CS 538, Spring 2019: HW5
 *
 * Binary Search Tree
 * See `bst.md` for the overview.
 */

#![allow(dead_code)]
#![forbid(unsafe_code)]

use std::cmp::Ordering;
use std::fmt::Debug;
use std::iter::{FromIterator, IntoIterator};
use std::ops::{Index, IndexMut};

/// Type of BST nodes.
#[derive(Debug)]
struct Node<K, V> {
    key: K,
    val: V,
    size: usize,
    lt: TreeMap<K, V>,
    rt: TreeMap<K, V>,
}

/// Type of BSTs.
///
/// See TMLL 2.1 for details about the memory layout if you are curious.
#[derive(Debug, Default)]
pub struct TreeMap<K, V> {
    inner: Option<Box<Node<K, V>>>,
}

/// Part 1: Basic operations and traits (50)
///
/// Your first task is to implement the following operations on BSTs.
///
/// The API here is modeled after Rust's standard BTreeMap collections type. The implementation
/// details are up to you, but you should not change the types of public functions.
impl<K, V> TreeMap<K, V>
where
    K: Ord + Debug,
    V: Debug,
{
    /// Make a new TreeMap.
    pub fn new() -> Self {
        let tree_map: TreeMap<K, V> = TreeMap { inner: None };

        tree_map
    }

    /// Clear a TreeMap.
    pub fn clear(&mut self) {
        self.inner = None;
    }

    /// Check if a TreeMap is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_none()
    }

    /// Compute the size of a TreeMap.
    pub fn len(&self) -> usize {
        let size: usize;
        match &self.inner {
            None => size = 0,
            Some(node) => size = node.size,
        };

        size
    }

    /// Check if a TreeMap has a certain key.
    pub fn has_key(&self, key: &K) -> bool {
        let ret_val: bool;
        match &self.inner {
            None => ret_val = false,
            Some(node) => match key.cmp(&node.key) {
                Ordering::Equal => ret_val = true,
                Ordering::Less => ret_val = node.lt.has_key(key),
                Ordering::Greater => ret_val = node.rt.has_key(key),
            },
        };

        ret_val
    }

    /// Get a reference to the value associated with a key, if present.
    ///
    /// If the key is not in the map, return None.
    pub fn get(&self, key: &K) -> Option<&V> {
        let ret_val: Option<&V>;
        if !self.has_key(key) {
            ret_val = None;
        } else {
            match &self.inner {
                None => {
                    println!("Error getting reference to value with key: {:?}\n Something probably wrong with fn: has_key", key);
                    ret_val = None;
                }
                Some(node) => match key.cmp(&node.key) {
                    Ordering::Equal => ret_val = Some(&node.val),
                    Ordering::Less => ret_val = node.lt.get(key),
                    Ordering::Greater => ret_val = node.rt.get(key),
                },
            };
        }

        ret_val
    }

    /// Get a mutable reference to the value associated with a key, if present.
    ///
    /// If the key is not in the map, return None.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let ret_val: Option<&mut V>;
        if !self.has_key(key) {
            ret_val = None;
        } else {
            match &mut self.inner {
                None => {
                    println!("Error getting reference to value with key: {:?}\n Something probably wrong with fn: has_key", key);
                    ret_val = None;
                }
                Some(node) => {
                    match key.cmp(&node.key) {
                        Ordering::Equal => ret_val = Some(&mut node.val),
                        Ordering::Less => ret_val = node.lt.get_mut(key),
                        Ordering::Greater => ret_val = node.rt.get_mut(key),
                    };
                }
            };
        }

        ret_val
    }

    /// Insert a (key, value) pair into a TreeMap.
    ///
    /// If the key is already present in the map, return the previous value and replace the old
    /// value with the new value. Otherwise, insert the new (key, value) pair and return None.
    pub fn insert(&mut self, key: K, val: V) -> Option<V> {
        let ret_val: Option<V>;
        if self.has_key(&key) {
            match &mut self.inner {
                None => {
                    println!("Error getting reference to value with key: {:?}\n Something probably wrong with fn: has_key", key);
                    ret_val = None;
                }
                Some(node) => {
                    match key.cmp(&node.key) {
                        Ordering::Equal => {
                            let old_val = std::mem::replace(&mut node.val, val);
                            ret_val = Some(old_val);
                        }
                        Ordering::Less => {
                            ret_val = node.lt.insert(key, val);
                        }
                        Ordering::Greater => {
                            ret_val = node.rt.insert(key, val);
                        }
                    };
                }
            };
        } else {
            match &mut self.inner {
                None => {
                    let new_node: Node<K, V> = Node {
                        key,
                        val,
                        size: 1,
                        lt: TreeMap::new(),
                        rt: TreeMap::new(),
                    };
                    self.inner = Some(Box::new(new_node));
                    ret_val = None;
                }
                Some(node) => {
                    match key.cmp(&node.key) {
                        Ordering::Equal => {
                            println!("Error getting reference to value with key: {:?}\n Something probably wrong with fn: has_key", key);
                            ret_val = None;
                        }
                        Ordering::Less => {
                            node.size += 1;
                            ret_val = node.lt.insert(key, val);
                        }
                        Ordering::Greater => {
                            node.size += 1;
                            ret_val = node.rt.insert(key, val);
                        }
                    };
                }
            };
        }

        ret_val
    }

    /// Insert a nonempty TreeMap into a TreeMap.
    ///
    /// While real BSTs do careful rotations/rebalancing to keep the depth of the merged tree as
    /// small as possible, we will just do a naive insert by going down to an appropriate leaf of
    /// `self` and sticking `other` there.
    ///
    /// Since this function is not publicly visible (there is no `pub` before `fn`), it is not used
    /// directly by clients. You may assume that `self` and `other` are non-overlapping: either all
    /// nodes in self are strictly smaller than all nodes in other, or all nodes in self are
    /// strictly larger than all nodes in other. Your implementation can panic (or just continue
    /// silently) if this requirement is not met. However if the trees are not overlapping, you
    /// must maintain the BST invariant.
    fn insert_tree(&mut self, mut other: Self) {
        match &mut self.inner {
            None => {
                self.inner = other.inner.take();
            }
            Some(node1) => {
                match &mut other.inner {
                    None => {
                        std::panic!("Error: trying to insert an empty tree");
                    }
                    Some(node2) => {
                        match &node2.key.cmp(&node1.key) {
                            Ordering::Equal => {
                                std::panic!("Error: trying to insert a tree. Invariant not met");
                            }
                            Ordering::Less => {
                                node1.size += node2.size;
                                node1.lt.insert_tree(other);
                            }
                            Ordering::Greater => {
                                node1.size += node2.size;
                                node1.rt.insert_tree(other);
                            }
                        };
                    }
                };
            }
        };
    }

    /// Remove a key from a TreeMap.
    ///
    /// If the map contains the key, remove the key and return the associated value.
    /// If the map does not contain the key, return None and leave the map unchanged.
    ///
    /// Hint: take the two child trees of the node you're removing, insert one into the other.
    pub fn remove(&mut self, key: K) -> Option<V> {
        let ret_val: Option<V> = if !self.has_key(&key) {
            None
        } else {
            // Match on ancestors of node with key K. Decrement size
            self.decrement_size_ancestors(key);
            // Remove the key. Join subtrees
            let mut removed_node = self.inner.take().unwrap();
            let left = &mut removed_node.lt;
            left.insert_tree(removed_node.rt);
            self.inner = std::mem::replace(&mut removed_node.lt.inner, None);
            Some(removed_node.val)
        };

        ret_val
    }

    fn decrement_size_ancestors(&mut self, key: K) {
        match &mut self.inner {
            None => {
                println!("Error getting reference to value with key: {:?}\n Something probably wrong with fn: has_key", key);
            }
            Some(node) => {
                node.size -= 1;
                if node.lt.has_key(&key) {
                    node.lt.decrement_size_ancestors(key);
                } else if node.rt.has_key(&key) {
                    node.rt.decrement_size_ancestors(key);
                } else {
                }
            }
        };
    }

    pub fn iter(&self) -> Iter<K, V> {
        Iter::new(self)
    }

    pub fn iter_mut(&mut self) -> IterMut<K, V> {
        IterMut::new(self)
    }
}

/// We will implement a few traits for TreeMap. Starting off: implement the FromIterator trait.
///
/// Hint: feed stuff from the iterator into the TreeMap.
impl<K, V> FromIterator<(K, V)> for TreeMap<K, V>
where
    K: Ord + Debug,
    V: Debug,
{
    fn from_iter<T>(iter: T) -> TreeMap<K, V>
    where
        T: IntoIterator<Item = (K, V)>,
    {
        let mut tree: TreeMap<K, V> = TreeMap::new();

        for item in iter {
            tree.insert(item.0, item.1);
        }

        tree
    }
}

/// Now, we will implement two kinds of indexing traits: Index and IndexMut.
///
/// Rust's built-in syntactic sugar hooks into these traits, for instance we can write stuff like:
///
/// let val = map[idx];
///
/// which is short for
///
/// let val = *map.index(idx);
///
/// if we have the Index trait. If we implement the IndexMut trait, we can write stuff like:
///
/// map[idx] = new_val;
///
/// which is short for
///
/// *map.index(idx) = new_val;
///
/// Index returns a reference to the value for a given key, and IndexMut returns a mutable
/// reference to the value for a given key. If the key is not in the map, panic. You will probably
/// want to take a look at the documentation for Index and IndexMut.
///
/// Note: the Rust BTreeMap actually has a more general type for these operations, something like:
///
/// fn index<Q>(&self, key: &Q) -> &V
/// where
///      K: Borrow<Q>,
///      Q: Ord + ?Sized,
/// { ... }
///
/// The idea here is that K is the key type stored in the map, while Q is the key type used for
/// lookups. In some cases, this allows us to avoid constructing a new object just to do a lookup;
/// think K = String and Q = &str. The trait bound K: Borrow<Q> says that if we have a K, we can
/// get a reference to Q. The Then, we can compare Q's since Q: Ord. (The question-mark bound Q:
/// ?Sized means that the size of Q does *not* need to be known at compile time; by default, all
/// type parameters have statically known sizes.) See the documentation for Borrow for an example
/// of how this works:
///
/// https://doc.rust-lang.org/std/borrow/trait.Borrow.html
impl<'a, K, V> Index<&'a K> for TreeMap<K, V>
where
    K: Ord + Debug,
    V: Debug,
{
    type Output = V;

    fn index(&self, key: &K) -> &V {
        let ret_val: &V;
        match &self.inner {
            None => std::panic!("Failed index"),
            Some(node) => {
                match key.cmp(&node.key) {
                    Ordering::Equal => {
                        ret_val = &node.val;
                    }
                    Ordering::Less => {
                        ret_val = node.lt.index(key);
                    }
                    Ordering::Greater => {
                        ret_val = node.rt.index(key);
                    }
                };
            }
        };

        ret_val
    }
}

impl<'a, K, V> IndexMut<&'a K> for TreeMap<K, V>
where
    K: Ord + Debug,
    V: Debug,
{
    fn index_mut(&mut self, key: &'a K) -> &mut V {
        let ret_val: &mut V;
        match &mut self.inner {
            None => std::panic!("Failed index"),
            Some(node) => {
                match key.cmp(&node.key) {
                    Ordering::Equal => {
                        ret_val = &mut node.val;
                    }
                    Ordering::Less => {
                        ret_val = node.lt.index_mut(key);
                    }
                    Ordering::Greater => {
                        ret_val = node.rt.index_mut(key);
                    }
                };
            }
        };

        ret_val
    }
}

/// Part 2: Iterators (35)
///
/// Now, you will soup up your BST by implementing various kinds of iterators to perform in-order
/// traversals of the map, i.e., from smallest to largest key.
///
/// The first iterator you will design is a so-called *consuming iterator*, which takes ownership
/// of all elements in the map. You will use the following types to track the current state of the
/// iterator:

#[derive(Debug)]
enum Next<I, T> {
    Item(I),
    Tree(T),
}

#[derive(Debug, Default)]
pub struct IntoIter<K, V> {
    next_nodes: Vec<Next<(K, V), TreeMap<K, V>>>,
    current_val: Option<(K, V)>,
}

/// This state has two main parts. There is the current key value pair (or nothing, if the iterator
/// is already finished), and stack of `Next` recording the next elements of the iterator.  For
/// instance, if the iterator is currently at a node that is a left-child of its parent, the top of
/// the stack would be the data at the parent, the 2nd-from-top of the stack would store the whole
/// subtree from the right sibling, and so on.
///
/// Define a few functions to set up the consuming iterator.
impl<K, V> IntoIter<K, V> {
    /// Make a new consuming iterator from a TreeMap
    ///
    /// Note that this function takes ownership (i.e., tree is passed by value). This is fine,
    /// because we are building a *consuming* iterator.
    fn new(tree: TreeMap<K, V>) -> Self {
        let mut nodes: Vec<Next<(K, V), TreeMap<K, V>>> = Vec::new();
        nodes.push(Next::Tree(tree));
        IntoIter {
            current_val: None,
            next_nodes: nodes,
        }
    }

    /// This is the main workhorse function for setting up the iterator. In words, this function
    /// should descend to the left-furthermost non-leaf child. Along the way, it sets up the
    /// iterator state: the current value, and the next-nodes-to-visit stack. Make sure you add
    /// nodes in the correct order: we are aiming for an in-order traversal, so the iterator should
    /// visit left-child, parent, right-child as we pop things off the stack.
    //fn descend_left(&mut self, tree: TreeMap<K, V>) {
    fn descend_left(&mut self, mut tree: TreeMap<K, V>) {
        match tree.inner.take() {
            Some(node) => {
                let left_child_tree = node.lt;
                let right_child_tree = node.rt;

                if left_child_tree.inner.is_none() {
                    // Leftest child
                    self.current_val = Some((node.key, node.val));
                    if right_child_tree.inner.is_some() {
                        self.next_nodes.push(Next::Tree(right_child_tree));
                    }
                } else {
                    self.next_nodes.push(Next::Tree(right_child_tree));
                    self.next_nodes.push(Next::Item((node.key, node.val)));
                    self.descend_left(left_child_tree);
                }
            }
            _ => {
                self.current_val = None;
            }
        };
    }
}

/// Implement IntoIterator for TreeMap.
impl<K, V> IntoIterator for TreeMap<K, V>
where
    K: Ord + Debug,
    V: Debug,
{
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> IntoIter<K, V> {
        IntoIter::new(self)
    }
}

/// Now, we are ready to implement the Iterator trait for IntoIter.
///
/// The main function `next` should get the current value, update the iterator to prepare the next
/// value, and then update the nodes to visit.
///
/// Hint: you'll want to use `descend_left`.
impl<K, V> Iterator for IntoIter<K, V>
where
    K: Debug,
    V: Debug,
{
    type Item = (K, V);
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_val.is_none() && self.next_nodes.len() == 1 {
            // Initialize current_val and next_nodes
            match self.next_nodes.pop() {
                Some(Next::Tree(tree)) => {
                    self.descend_left(tree);
                }
                _ => {}
            };
        }

        // Process current_val. Set up for return
        let ret_val: Option<Self::Item>;
        match &mut self.current_val {
            None => ret_val = None,
            Some(_) => {
                ret_val = std::mem::replace(&mut self.current_val, None);
            }
        };

        // Set up current_val and next_nodes for future next() call
        if !self.next_nodes.is_empty() {
            match self.next_nodes.pop() {
                Some(Next::Item(item)) => {
                    self.current_val = Some(item);
                }
                Some(Next::Tree(tree)) => {
                    self.descend_left(tree);
                }
                _ => {}
            };
        }

        ret_val
    }
}

/// Now, you'll define a few variants of iterators that do not consume the data. These are also
/// called *borrowing* iterators, since they borrow data instead of consuming it. The overall
/// strategy will be quite similar to what you already did above, but with a few tweaks. The main
/// idea behind a borrowing iterator is that we should produce references to data, rather than data.
///
/// For this part, you'll be designing the type structures and implementing the necessary functions.
/// You should design two kinds of iterators:
///
/// 1. Borrowing iterator. This should implement the Iterator trait with item type (&K, &V).
/// 2. Mutable iterator. This should implement the Iterator trait with item type (&K, &mut V).
///
/// Note that we need to be a bit careful with mutable iterators: clients should be allowed to
/// modify the value, but they should not be allowed to modify the keys---if they did, they might
/// screw up the BST property. Your mutable iterator should hand out a pair of references: an
/// *immutable* reference to the key, and a *mutable* reference to the value.
///
/// To construct these two iterators, implement the following two methods for TreeMap:
///
/// 1. pub fn iter(&self) -> Iter<K, V>
/// 2. pub fn iter_mut(&mut self) -> IterMut<K, V>
///
/// Iter and IterMut should implement the Iterator trait.
///
/// You should also implement versions of all the associated functions you did for IntoIter, but do
/// not implement IntoIterator for TreeMap again.
///
/// Hint: don't implement these iterators from scratch. You'll need to sprinkle in some extra
/// annotations and tweaks in a few key places, but your code for the borrowing and mutable
/// iterators should be a nearly exact copy-paste of your code for consuming iterators.

#[derive(Debug)]
pub struct Iter<'a, K, V> {
    next_nodes: Vec<Next<(&'a K, &'a V), &'a TreeMap<K, V>>>,
    current_val: Option<(&'a K, &'a V)>,
}

impl<'a, K, V> Iter<'a, K, V> {
    fn new(tree: &'a TreeMap<K, V>) -> Self {
        let mut nodes: Vec<Next<(&'a K, &'a V), &'a TreeMap<K, V>>> = Vec::new();
        nodes.push(Next::Tree(tree));
        Iter {
            current_val: None,
            next_nodes: nodes,
        }
    }

    fn descend_left(&mut self, tree: &'a TreeMap<K, V>) {
        match &tree.inner {
            Some(node) => {
                let left_child_tree = &node.lt;
                let right_child_tree = &node.rt;

                if left_child_tree.inner.is_none() {
                    // Leftest child
                    self.current_val = Some((&node.key, &node.val));
                    if right_child_tree.inner.is_some() {
                        self.next_nodes.push(Next::Tree(right_child_tree));
                    }
                } else {
                    self.next_nodes.push(Next::Tree(right_child_tree));
                    self.next_nodes.push(Next::Item((&node.key, &node.val)));
                    self.descend_left(left_child_tree);
                }
            }
            _ => {
                self.current_val = None;
            }
        };
    }
}

impl<'a, K, V> Iterator for Iter<'a, K, V>
where
    K: Debug,
    V: Debug,
{
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_val.is_none() && self.next_nodes.len() == 1 {
            // Initialize current_val and next_nodes
            match self.next_nodes.pop() {
                Some(Next::Tree(tree)) => {
                    self.descend_left(tree);
                }
                _ => {}
            };
        }

        // Process current_val. Set up for return
        let ret_val: Option<Self::Item>;
        match &mut self.current_val {
            None => ret_val = None,
            Some(_) => {
                ret_val = std::mem::replace(&mut self.current_val, None);
            }
        };

        // Set up current_val and next_nodes for future next() call
        if !self.next_nodes.is_empty() {
            match self.next_nodes.pop() {
                Some(Next::Item(item)) => {
                    self.current_val = Some(item);
                }
                Some(Next::Tree(tree)) => {
                    self.descend_left(tree);
                }
                _ => {}
            };
        }

        ret_val
    }
}

#[derive(Debug)]
pub struct IterMut<'a, K, V> {
    next_nodes: Vec<Next<(&'a K, &'a mut V), &'a mut TreeMap<K, V>>>,
    current_val: Option<(&'a K, &'a mut V)>,
}

impl<'a, K, V> IterMut<'a, K, V> {
    fn new(tree: &'a mut TreeMap<K, V>) -> Self {
        let mut nodes: Vec<Next<(&'a K, &'a mut V), &'a mut TreeMap<K, V>>> = Vec::new();
        nodes.push(Next::Tree(tree));
        IterMut {
            current_val: None,
            next_nodes: nodes,
        }
    }

    fn descend_left(&mut self, tree: &'a mut TreeMap<K, V>) {
        match &mut tree.inner {
            Some(node) => {
                let left_child_tree = &mut node.lt;
                let right_child_tree = &mut node.rt;

                if left_child_tree.inner.is_none() {
                    // Leftest child
                    self.current_val = Some((&node.key, &mut node.val));
                    if right_child_tree.inner.is_some() {
                        self.next_nodes.push(Next::Tree(right_child_tree));
                    }
                } else {
                    self.next_nodes.push(Next::Tree(right_child_tree));
                    self.next_nodes.push(Next::Item((&node.key, &mut node.val)));
                    self.descend_left(left_child_tree);
                }
            }
            _ => {
                self.current_val = None;
            }
        };
    }
}

impl<'a, K, V> Iterator for IterMut<'a, K, V>
where
    K: Debug,
    V: Debug,
{
    type Item = (&'a K, &'a mut V);
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_val.is_none() && self.next_nodes.len() == 1 {
            // Initialize current_val and next_nodes
            match self.next_nodes.pop() {
                Some(Next::Tree(tree)) => {
                    self.descend_left(tree);
                }
                _ => {}
            };
        }

        // Process current_val. Set up for return
        let ret_val: Option<Self::Item>;
        match &mut self.current_val {
            None => ret_val = None,
            Some(_) => {
                ret_val = std::mem::replace(&mut self.current_val, None);
            }
        };

        // Set up current_val and next_nodes for future next() call
        if !self.next_nodes.is_empty() {
            match self.next_nodes.pop() {
                Some(Next::Item(item)) => {
                    self.current_val = Some(item);
                }
                Some(Next::Tree(tree)) => {
                    self.descend_left(tree);
                }
                _ => {}
            };
        }

        ret_val
    }
}

/// Part 3: Custom dropping (15)
///
/// The built-in Drop trait behaves poorly because it drops TreeMap recursively.  If the TreeMap is
/// too deep, this will blow up the stack when dropping (see TMLL 2.7 for details).
///
/// Implement a custom Drop trait to drop the TreeMap in a loop rather than recursing. The default
/// Drop implementation for the TreeMap/Node types drops according to a *pre-order* traversal: drop
/// root, then drop left, then drop right. Your implementation should maintain this order.
impl<K, V> Drop for TreeMap<K, V> {
    fn drop(&mut self) {
        let mut node_stack: Vec<Option<Box<Node<K, V>>>> = Vec::new();
        node_stack.push(self.inner.take());
        while !node_stack.is_empty() {
            match &mut node_stack.pop().unwrap() {
                None => {}
                Some(node) => {
                    node_stack.push(node.rt.inner.take());
                    node_stack.push(node.lt.inner.take());
                }
            };
        }
    }
}

#[cfg(test)]
mod test {
    use super::TreeMap;
    use std::fmt::Debug;

    impl<K, V> TreeMap<K, V>
    where
        K: Ord + Debug,
        V: Debug,
    {
        /// Test function: checks the BST invariant.
        fn is_bst(&self) -> bool {
            let mut lsize = 0;
            let mut rsize = 0;
            if let Some(boxed) = &self.inner {
                let lt = &boxed.lt;
                let rt = &boxed.rt;

                if let Some(lnode) = &lt.inner {
                    if lnode.key >= boxed.key || !lt.is_bst() {
                        return false;
                    }

                    lsize = lnode.size;
                }

                if let Some(rnode) = &rt.inner {
                    if rnode.key <= boxed.key || !rt.is_bst() {
                        return false;
                    }

                    rsize = rnode.size;
                }

                return boxed.size == lsize + rsize + 1;
            }

            true
        }
    }

    #[test]
    fn test_insert() {
        let mut tree = TreeMap::new();

        tree.insert(1, 11);
        tree.insert(3, 33);
        tree.insert(2, 22);

        assert_eq!(tree.get(&1), Some(&11));
        assert_eq!(tree.get(&2), Some(&22));
        assert_eq!(tree.get(&3), Some(&33));

        assert!(tree.is_bst());

        let insert_res = tree.insert(1, 111);

        assert_eq!(tree.get(&1), Some(&111));
        assert_eq!(insert_res, Some(11));

        assert!(tree.is_bst());
    }

    #[test]
    fn test_clear() {
        let mut tree = TreeMap::new();

        tree.insert(1, 11);
        tree.insert(3, 33);
        tree.insert(2, 22);

        assert_eq!(tree.len(), 3);

        tree.clear();

        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn test_remove() {
        let mut tree = TreeMap::new();

        tree.insert(1, 11);
        tree.insert(3, 33);
        tree.insert(2, 22);

        assert_eq!(tree.remove(1), Some(11));
        assert_eq!(tree.len(), 2);
        assert!(tree.is_bst());

        assert_eq!(tree.remove(1), None);
        assert_eq!(tree.len(), 2);
        assert!(tree.is_bst());
    }

    #[test]
    fn test_mut() {
        let mut tree = TreeMap::new();

        tree.insert(1, 11);
        tree.insert(3, 33);
        tree.insert(2, 22);

        assert_eq!(tree.get(&3), Some(&33));

        *tree.get_mut(&3).unwrap() = 333;

        assert_eq!(tree.get(&3), Some(&333));

        assert!(tree.is_bst());
    }

    #[test]
    fn test_index() {
        let mut tree = TreeMap::new();

        tree.insert(1, 11);
        tree.insert(3, 33);
        tree.insert(2, 22);

        assert_eq!(tree[&1], 11);

        tree[&2] = 22;

        assert_eq!(tree.get(&2), Some(&22));

        assert!(tree.is_bst());
    }

    #[should_panic]
    #[test]
    fn test_bad_index() {
        let mut tree = TreeMap::new();

        tree.insert(1, 11);
        tree.insert(3, 33);
        tree.insert(2, 22);

        tree[&5] = 10;
    }

    #[test]
    fn test_from_iter() {
        let vec = vec![(1, 11), (3, 33), (2, 22)];
        let tree: TreeMap<i32, i32> = vec.into_iter().collect();

        assert!(tree.is_bst());

        assert_eq!(tree.get(&1), Some(&11));
        assert_eq!(tree.get(&2), Some(&22));
        assert_eq!(tree.get(&3), Some(&33));
    }

    #[test]
    fn test_iter() {
        let vec = vec![(1, 11), (3, 33), (2, 22)];
        let tree: TreeMap<i32, i32> = vec.into_iter().collect();

        let mut iter = tree.into_iter();

        assert_eq!(iter.next(), Some((1, 11)));
        assert_eq!(iter.next(), Some((2, 22)));
        assert_eq!(iter.next(), Some((3, 33)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_borrow_iter() {
        let vec = vec![(1, 11), (3, 33), (2, 22)];
        let tree: TreeMap<i32, i32> = vec.into_iter().collect();

        let mut iter = tree.iter();

        assert_eq!(iter.next(), Some((&1, &11)));
        assert_eq!(iter.next(), Some((&2, &22)));
        assert_eq!(iter.next(), Some((&3, &33)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_mut_iter() {
        let vec = vec![(1, 11), (3, 33), (2, 22)];
        let mut tree: TreeMap<i32, i32> = vec.into_iter().collect();

        for (k, v) in tree.iter_mut() {
            *v = k + 100;
        }

        assert!(tree.is_bst());
        assert_eq!(tree.len(), 3);

        assert_eq!(tree.get(&1), Some(&101));
        assert_eq!(tree.get(&2), Some(&102));
        assert_eq!(tree.get(&3), Some(&103));
    }

    use std::cell::RefCell;

    #[derive(Debug)]
    struct MyDropper<'a> {
        inner: String,
        log: &'a RefCell<Vec<String>>,
    }

    impl<'a> Drop for MyDropper<'a> {
        fn drop(&mut self) {
            // Record the dropped String in a shared log
            let newstr = self.inner.clone();
            self.log.borrow_mut().push(newstr);
        }
    }

    #[test]
    fn test_drop() {
        let drops = RefCell::new(vec![]);

        // Contents of tree in pre-order traversal
        let contents = vec![
            String::from("m"),
            String::from("h"),
            String::from("a"),
            String::from("k"),
            String::from("t"),
            String::from("p"),
            String::from("z"),
        ];

        // Set up a complete binary tree (key = val)
        {
            let mut tree = TreeMap::new();
            for s in &contents {
                let log = &drops;
                let key = s.clone();
                let inner = s.clone();
                tree.insert(key, MyDropper { inner, log });
            }
        } // ... and drop it

        // Check the order of drops
        assert_eq!(*drops.borrow(), contents);
    }

    /* Uncomment when drop trait ready.
    #[test]
    fn test_drop_big() {
        // This is slow (~1-2 min on my machine) but it shouldn't overflow!
        let v = (1..100000).collect::<Vec<u32>>();

        // Make a really deep binary tree
        {
            let mut tree = TreeMap::new();
            for i in v {
                tree.insert(i, i);
            }
        } // ... and drop it

        println!("All done!");
    } */
}
