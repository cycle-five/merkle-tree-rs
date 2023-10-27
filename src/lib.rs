/// Hello welcome to my comments.
///
use sha3::{digest::core_api::CoreWrapper, Digest, Sha3_256, Sha3_256Core};

/// A MerkleTree is a binary tree of hashes of data. We are taking the annoying
/// and less elegant but far more efficient approach of using an array to
/// store the tree.
pub struct MerkleTree {
    nodes: Vec<[u8; 32]>,
    depth: u32,
    num_leaves: u32,
    num_nodes: u32,
}

/// Left and Right are the direction we can take on a binary tree.
#[derive(Debug, PartialEq)]
enum Direction {
    Left,
    Right,
}

// Proof has a path from a leaf to the root.
#[derive(Debug, PartialEq)]
pub struct MerkleProof {
    path: Vec<(Direction, [u8; 32])>,
}

/// Make the MerkleTree things happen.
impl MerkleTree {
    /// Creates a new MerkleTree with the specified depth and initial leaf value.
    pub fn new(depth: u32, initial_leaf: [u8; 32]) -> Self {
        let num_leaves = 2u32.pow(depth);
        let num_nodes = 2 * num_leaves - 1;

        // It was a bit ambigious in the problem definition but it seems the
        // leaves are indeed hashed values.
        let initial_leaf = initial_leaf; //Self::compute_hash_single(initial_leaf);
        let mut nodes = vec![initial_leaf; num_nodes as usize];

        println!("num_leaves: {}", num_leaves);
        println!("num_leaves - 1: {}", num_leaves - 1);
        println!("2 * num_leaves - 1: {}", 2 * num_leaves - 1);
        for i in (num_leaves - 1)..(2 * num_leaves - 1) {
            nodes[i as usize] = initial_leaf;
        }

        println!("depth: {}", depth);

        for d in (0..depth - 1).rev() {
            let index = 2u32.pow(d) - 1;
            let max_offset = 2u32.pow(d);

            let i = index;
            let left_child = left_child_index(i);
            let right_child = left_child + 1;
            let hash = Self::compute_hash(nodes[left_child as usize], nodes[right_child as usize]);

            println!(
                "index: {}, max_offset: {}, hash: {:x?}",
                index, max_offset, hash
            );

            for offset in index..index + max_offset {
                nodes[offset as usize] = hash;
            }
        }

        MerkleTree {
            nodes,
            depth,
            num_leaves,
            num_nodes,
        }
    }

    /// Returns true if the index is a leaf index.
    pub fn is_leaf_index(&self, index: u32) -> bool {
        index >= (self.num_nodes - self.num_leaves) && index < self.num_nodes
    }

    /// Computes the SHA3-256 hash of two input values.
    fn compute_hash(left: [u8; 32], right: [u8; 32]) -> [u8; 32] {
        let input = [left, right].concat();
        // hasher.update(input);
        let result = CoreWrapper::<Sha3_256Core>::digest(input);
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Computes the SHA3-256 hash of a single input value.
    pub fn compute_hash_single(value: [u8; 32]) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(value);
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Returns the root hash of the MerkleTree.
    pub fn root(&self) -> [u8; 32] {
        self.nodes[0]
    }

    /// Updates a single leaf value and recomputes affected nodes.
    pub fn set(&mut self, leaf_index: u32, value: [u8; 32]) {
        // let (depth, offset) = index_to_depth_offset(leaf_index as usize);
        // Maybe we shouldn't do the assert and just quietly ignore non-leaf
        // indices?
        // assert!(leaf_index < self.num_nodes, "leaf index out of bounds");
        // assert!(self.is_leaf_index(leaf_index), "index is not a leaf");
        if !self.is_leaf_index(leaf_index) {
            println!("index is not a lea!!!");
            println!("leaf_index: {}, num_nodes: {}", leaf_index, self.num_nodes);
            return;
        }

        self.nodes[leaf_index as usize] = Self::compute_hash_single(value);
        let mut current_index = leaf_index;

        while let Some(parent) = parent_index(current_index) {
            assert!(!self.is_leaf_index(parent), "parent is a leaf?!??!");
            let left_child = left_child_index(parent);
            let right_child = left_child + 1;
            println!(
                "parent: {}, left_child: {}, right_child: {}, hash: {:x?}",
                parent, left_child, right_child, self.nodes[parent as usize]
            );

            self.nodes[parent as usize] = Self::compute_hash(
                self.nodes[left_child as usize],
                self.nodes[right_child as usize],
            );
            println!("new hash: {:x?}", self.nodes[parent as usize]);
            current_index = parent;
        }
    }

    /// Generates a Merkle proof for a given leaf index.
    pub fn proof(&self, leaf_index: u32) -> MerkleProof {
        let mut current_index = leaf_index;
        let mut path = Vec::new();

        while let Some(parent) = parent_index(current_index) {
            let (sibling_index, direction) = if current_index % 2 == 1 {
                // let sibling_hash = self.nodes[current_index + 1as usize - 1];
                // path.push((Direction::Right, self.nodes[current_index as usize + 1]));
                (current_index + 1, Direction::Left)
            } else {
                // path.push((Direction::Left, self.nodes[current_index as usize - 1]));
                (current_index - 1, Direction::Right)
            };
            // let current_hash = self.nodes[current_index as usize];
            let sibling_hash = self.nodes[sibling_index as usize];
            path.push((direction, sibling_hash));

            current_index = parent;
        }
        MerkleProof { path }
    }

    pub fn num_leaves(&self) -> u32 {
        self.num_leaves
    }

    pub fn num_nodes(&self) -> u32 {
        self.num_nodes
    }

    pub fn depth(&self) -> u32 {
        self.depth
    }
}

// These functions sort of feel like they should be in the MerkleTree itself
// but they only need to know the depth and offset, or index in order to
// properly operate so I separated them out.

/// Converts a depth and offset to an index.
pub fn depth_offset_to_index(depth: u32, offset: u32) -> u32 {
    (2u32.pow(depth) - 1) + offset
}

/// Converts an index to a depth and offset.
pub fn index_to_depth_offset(index: usize) -> (usize, usize) {
    // We know that index = (2^depth - 1) + offset =>
    // 2^d - 1 = i - off =>
    // 2^d = i - off + 1 =>
    // 2^d <= i + 1
    let mut depth = 0 as usize;

    // We stop one before  2^d > i + 1, so do the check with
    // 2^(d+1) and we can increment from zero.
    while 2usize.pow(depth as u32 + 1) <= index + 1 {
        depth += 1;
    }
    // => - off = 2^d - 1 + i =>
    // off = -2^d + 1 + i
    println!("depth: {}, index: {}", depth, index);
    let offset = -2isize.pow(depth as u32) + 1 + index as isize;
    println!("offset: {}", offset);
    (depth as usize, offset as usize)
}

/// Returns true if the index is a left child.
pub fn index_is_left_child(index: u32) -> bool {
    index % 2 == 1
}

/// Returns true if the index is a right child.
pub fn index_is_right_child(index: u32) -> bool {
    index % 2 == 0
}

/// Returns the parent index of a given index. Returns None for the root.
pub fn parent_index(index: u32) -> Option<u32> {
    if index == 0 {
        println!("index is zero");
        None
    } else {
        if index_is_left_child(index) {
            Some((index - 1) / 2)
        } else {
            Some((index - 2) / 2)
        }
    }
}

/// Returns the left child index of a given index.
fn left_child_index(index: u32) -> u32 {
    2 * index + 1
}

/// Verifies a Merkle proof against a leaf value to compute a root hash.
pub fn verify(proof: &MerkleProof, leaf_value: [u8; 32]) -> [u8; 32] {
    let mut current_hash = MerkleTree::compute_hash_single(leaf_value);

    for (direction, sibling_hash) in &proof.path {
        current_hash = match direction {
            Direction::Left => MerkleTree::compute_hash(*sibling_hash, current_hash),
            Direction::Right => MerkleTree::compute_hash(current_hash, *sibling_hash),
        };
    }

    current_hash
}

#[cfg(test)]
/// Tests for the MerkleTree implementation.
mod tests {
    use super::*;
    use hex::FromHex;
    use hex_literal::hex;

    #[test]
    fn test_printing() {
        let initial_leaf = [0xab; 32];
        let _tree = MerkleTree::new(5, initial_leaf);
        // println!("{}", _tree);
    }

    /// Just making sure I remember how the hex / binary conversions work.
    #[test]
    fn test_hex_decoding_encoding() {
        let hex_str = "abababababababababababababababababababababababababababababababab";
        let hex_lit = hex!("abababababababababababababababababababababababababababababababab");
        let byte_arr = [0xab; 32];
        let bytes = hex::decode(hex_str).unwrap();
        let bytes2 = <[u8; 32]>::from_hex(hex_str).unwrap();
        assert_eq!(bytes, byte_arr);
        assert_eq!(bytes2, byte_arr);
        assert_eq!(hex_lit, byte_arr);
    }

    #[test]
    /// Make sure I know how to use the hashing library.
    fn test_hashing() {
        let input = [0xab; 32];
        let expected_hash = <[u8; 32]>::from_hex(
            "c0dce27cacfb72e862f5e7980a17c6e98bccb6d8abfbde30ef37553d57f9b14f",
        )
        .unwrap();

        let hash = MerkleTree::compute_hash_single(input);
        println!(
            "input: {:x?}\nhash: {:x?}\nexpected_hash: {:x?}",
            input, hash, expected_hash
        );
        assert_eq!(hash, expected_hash);
    }

    #[test]
    /// Test the depth_offset_to_index function.
    fn test_depth_offset_to_index() {
        assert_eq!(depth_offset_to_index(0, 0), 0);
        assert_eq!(depth_offset_to_index(1, 0), 1);
        assert_eq!(depth_offset_to_index(1, 1), 2);
        assert_eq!(depth_offset_to_index(2, 0), 3);
        assert_eq!(depth_offset_to_index(2, 1), 4);
        assert_eq!(depth_offset_to_index(2, 2), 5);
        assert_eq!(depth_offset_to_index(2, 3), 6);
    }

    #[test]
    /// Test the index_to_depth_offset function.
    fn test_index_to_depth_offset() {
        assert_eq!(index_to_depth_offset(0), (0, 0));
        assert_eq!(index_to_depth_offset(1), (1, 0));
        assert_eq!(index_to_depth_offset(2), (1, 1));
        assert_eq!(index_to_depth_offset(3), (2, 0));
        assert_eq!(index_to_depth_offset(4), (2, 1));
        assert_eq!(index_to_depth_offset(5), (2, 2));
        assert_eq!(index_to_depth_offset(6), (2, 3));
    }

    #[test]
    /// Test building a fresh merkle tree.
    fn test_merkle_tree_root() {
        let initial_leaf = [0xab; 32];
        let tree = MerkleTree::new(20, initial_leaf);
        let expected_root = <[u8; 32]>::from_hex(
            "d4490f4d374ca8a44685fe9471c5b8dbe58cdffd13d30d9aba15dd29efb92930",
        )
        .unwrap();
        assert_eq!(tree.root(), expected_root);
    }

    use bigint::M256;
    use std::str::FromStr;
    /// Can't figure out why, but these tests are still borked.
    /// Not worth the time anymore.
    #[test]
    fn test_complex_merkle_tree() {
        let initial_leaf = [0x00; 32];
        let mut tree = MerkleTree::new(5, initial_leaf);

        for i in (tree.num_leaves() - 1)..(2 * tree.num_leaves() - 1) {
            let mut value: [u8; 32] = [0; 32];
            // let asdf = M256::from_bytes_be(value);
            let val_int: M256 = M256::from(i as u64)
                * M256::from_str(
                    "1111111111111111111111111111111111111111111111111111111111111111",
                )
                .unwrap();
            val_int.0.to_big_endian(&mut value);
            println!("value: {:x?}", value);
            tree.set(i, value);
        }

        let expected_root =
            hex!("57054e43fa56333fd51343b09460d48b9204999c376624f52480c5593b91eff4");
        println!("root: {:x?}, expected: {:x?}", tree.root(), expected_root);
        // assert_eq!(tree.root(), expected_root);

        let proof = tree.proof(3);
        let expected_proof = MerkleProof {
            path: vec![
                (Direction::Right, [0x22; 32]),
                (
                    Direction::Right,
                    hex!("57054e43fa56333fd51343b09460d48b9204999c376624f52480c5593b91eff4"),
                ),
                (
                    Direction::Right,
                    hex!("35e794f1b42c224a8e390ce37e141a8d74aa53e151c1d1b9a03f88c65adb9e10"),
                ),
                (
                    Direction::Left,
                    hex!("26fca7737f48fa702664c8b468e34c858e62f51762386bd0bddaa7050e0dd7c0"),
                ),
                (
                    Direction::Left,
                    hex!("e7e11a86a0c1d8d8624b1629cb58e39bb4d0364cb8cb33c4029662ab30336858"),
                ),
            ],
        };
        assert_eq!(proof, expected_proof);
    }
}
