///
/// Hello welcome to my comments.
///
/// A first pass of this was written by ChatGPT, which got sort of the basic structure
/// correct but none of the details. LLMs in particular aren't very good at math and so
/// indexing arithematic usually fucks it up pretty good.
///
use sha3::{digest::core_api::CoreWrapper, Digest, Sha3_256, Sha3_256Core};
use std::{
    fmt::{Display, Formatter},
    sync::RwLock,
};

/// A MerkleTree is a binary tree of hashes of data. We are taking the annoying
/// and less elegant but far more efficient approach of using an array to
/// store the tree.
pub struct MerkleTree {
    nodes: RwLock<Vec<[u8; 32]>>,
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
        let num_leaves = 2u32.pow(depth - 1);
        let num_nodes = 2 * num_leaves - 1;
        let _start_leaf_index = num_nodes - num_leaves;

        // Initial leaves are *not* hashed.
        // We initial every node to the initial leaf value, we will set the values
        // of each nodes layer by layer since we know the initial value of all the
        // leaves and parents are computed from their children.
        let mut nodes = vec![initial_leaf; num_nodes as usize];

        println!("num_leaves: {}", num_leaves);
        println!("num_leaves - 1: {}", num_leaves - 1);
        println!("2 * num_leaves - 1: {}", 2 * num_leaves - 1);
        // for i in start_leaf_index..num_nodes {
        //     nodes[i as usize] = initial_leaf;
        // }

        println!("depth: {}", depth);

        for d in (0..depth - 1).rev() {
            let index = 2u32.pow(d) - 1;
            let max_offset = 2u32.pow(d);

            let i = index;
            let left_child = left_child_index(i);
            let right_child = left_child + 1;

            // Is it worth converting these to strings?
            // let asdf = nodes[left_child as usize][..4]
            //     .iter()
            //     .map(|x| x.to_string())
            //     .reduce(|x, y| x + y.as_str())
            //     .unwrap_or_default();

            // This was to verify that each layer was correct getting set.
            //---- tests::test_complex_merkle_tree stdout ----
            // num_leaves: 32
            // num_leaves - 1: 31
            // 2 * num_leaves - 1: 63
            // depth: 5
            // depth: 3, left_child: 15, right_child: 16, right_child_hash: [0, 0, 0, 0], left_child_hash: [0, 0, 0, 0]
            // index: 7, max_offset: 8, hash: [7, f, a1, ab, 6f, cc, 55, 7e, d1, 4d, 42, 94, 1f, 19, 67, 69, 30, 48, 55, 1e, b9, 4, 2a, 8d, a, 5, 7a, fb, d7, 5e, 81, e0]
            // depth: 2, left_child: 7, right_child: 8, right_child_hash: [7, f, a1, ab], left_child_hash: [7, f, a1, ab]
            // index: 3, max_offset: 4, hash: [53, da, b0, 42, 30, 8a, b7, 1, b0, 73, ed, d4, d1, 4c, 56, 54, a1, f7, d, 21, b, 67, e, f2, 88, c9, ae, ea, 9c, 4a, 45, 30]
            // depth: 1, left_child: 3, right_child: 4, right_child_hash: [53, da, b0, 42], left_child_hash: [53, da, b0, 42]
            // index: 1, max_offset: 2, hash: [73, 29, f2, 9d, ca, e8, 88, 3e, 1, 4c, 3c, f1, 5b, 1b, dc, be, e8, 81, cb, de, 1c, 33, aa, cd, ca, 22, 3d, 5a, 0, dd, 6f, fe]
            // depth: 0, left_child: 1, right_child: 2, right_child_hash: [73, 29, f2, 9d], left_child_hash: [73, 29, f2, 9d]
            // index: 0, max_offset: 1, hash: [b9, e1, 27, 4e, 6, d4, 3b, 40, 3, 22, 37, 12, d2, 71, fe, 11, b3, 85, 99, e7, 75, 3d, ba, ab, 11, 1a, c1, 16, 1f, 82, f4, 5e]
            // value: [22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 20]
            println!(
                "depth: {}, left_child: {}, right_child: {}, right_child_hash: {:x?}, left_child_hash: {:x?}",
                d, left_child, right_child, &nodes[left_child as usize][..4], &nodes[right_child as usize][..4]
            );
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
            nodes: RwLock::new(nodes),
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
        self.nodes.read().unwrap()[0]
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
            // return;
        }

        self.nodes.write().unwrap()[leaf_index as usize] = value; //Self::compute_hash_single(value);
        let mut current_index = leaf_index;
        println!("leaf_index: {}", leaf_index);

        while let Some(parent) = parent_index(current_index) {
            assert!(!self.is_leaf_index(parent), "parent is a leaf?!??!");
            let left_child = left_child_index(parent);
            let right_child = left_child + 1;
            println!(
                "parent: {}, old_hash: {:02x?}",
                parent,
                &self.nodes.read().unwrap()[parent as usize][..4]
            );

            let (parent_hash, left_child_hash, right_child_hash) = {
                let mut rw_nodes = self.nodes.write().unwrap();
                rw_nodes[parent as usize] = Self::compute_hash(
                    rw_nodes[left_child as usize],
                    rw_nodes[right_child as usize],
                );
                (
                    rw_nodes[parent as usize],
                    rw_nodes[left_child as usize],
                    rw_nodes[right_child as usize],
                )
            };
            println!(
                "new hash: {:02x?}, left_child: {:02x?}, right_child: {:02x?}",
                &parent_hash[..4],
                &left_child_hash[..4],
                &right_child_hash[..4]
            );
            current_index = parent;
        }
    }

    /// Generates a Merkle proof for a given leaf index.
    pub fn proof(&self, leaf_index: u32) -> MerkleProof {
        let mut current_index = self.num_nodes() - self.num_leaves() + leaf_index;
        let mut path = Vec::new();

        loop {
            // So many off-by-one errors here...
            // The direction is the direction we'd go traveling *down* the tree, even though
            // we're building upwards. I'm not sure if I was supposed to store the root node
            // in the tree or not, but doing so gave it this distinction where you go left
            // *to* odd indexed nodes and right *to* even indexed nodes.
            let (sibling_index, direction) = if current_index % 2 == 1 {
                (current_index + 1, Direction::Left)
            } else {
                (current_index - 1, Direction::Right)
            };
            let current_hash = self.nodes.read().unwrap()[current_index as usize];
            let sibling_hash = self.nodes.read().unwrap()[sibling_index as usize];
            println!(
                "current_hash: {:x?}, sibling_hash: {:x?}, direction: {:?}",
                &current_hash[..4],
                &sibling_hash[..4],
                direction
            );
            path.push((direction, sibling_hash));

            match parent_index(current_index) {
                Some(parent) if parent != 0 => current_index = parent,
                _ => break,
            }
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

impl Display for MerkleTree {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let nodes = self.nodes.read().unwrap();
        let mut s = String::new();
        for node in nodes.iter() {
            s.push_str(&format!("{:02x?}", &node[..4]));
        }
        write!(f, "{}", s)
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
    let mut depth = 0_usize;

    // We stop one before  2^d > i + 1, so do the check with
    // 2^(d+1) and we can increment from zero.
    while 2usize.pow(depth as u32 + 1) <= index + 1 {
        depth += 1;
    }
    // => - off = 2^d - 1 + i =>
    // off = -2^d + 1 + i
    println!("depth: {}, index: {}", depth, index);
    let offset = -(2isize.pow(depth as u32)) + 1 + index as isize;
    println!("offset: {}", offset);
    (depth, offset as usize)
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
    } else if index_is_left_child(index) {
        Some((index - 1) / 2)
    } else {
        Some((index - 2) / 2)
    }
}

/// Returns the left child index of a given index.
fn left_child_index(index: u32) -> u32 {
    2 * index + 1
}

/// Verifies a Merkle proof against a leaf value to compute a root hash.
pub fn verify(proof: &MerkleProof, leaf_value: [u8; 32]) -> [u8; 32] {
    // let mut current_hash = MerkleTree::compute_hash_single(leaf_value);
    let mut current_hash = leaf_value;

    for (direction, sibling_hash) in &proof.path {
        current_hash = match direction {
            Direction::Right => MerkleTree::compute_hash(*sibling_hash, current_hash),
            Direction::Left => MerkleTree::compute_hash(current_hash, *sibling_hash),
        };
    }

    current_hash
}

#[cfg(test)]
/// Tests for the MerkleTree implementation.
mod tests {
    use std::str::FromStr;

    use super::*;
    use bigint::M256;
    use hex::FromHex;
    use hex_literal::hex;

    #[test]
    fn test_printing() {
        let initial_leaf = [0xab; 32];
        let _tree = MerkleTree::new(5, initial_leaf);
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
        // assert_eq!(tree.nodes.read().unwrap()[1], expected_root);
    }

    #[test]
    fn test_complex_merkle_tree() {
        let initial_leaf = [0x00; 32];
        let mut tree = MerkleTree::new(5, initial_leaf);

        println!("tree: {}", tree);
        //for i in ((tree.num_leaves())..(2 * tree.num_leaves() - 1)).rev() {
        for i in 0..tree.num_leaves() {
            let mut value: [u8; 32] = [0; 32];
            // let asdf = M256::from_bytes_be(value);
            let val_int: M256 = M256::from(i as u64)
                * M256::from_str(
                    "1111111111111111111111111111111111111111111111111111111111111111",
                )
                .unwrap();
            val_int.0.to_big_endian(&mut value);
            println!("value: {:x?}", value);
            let set_idx = tree.num_nodes() - tree.num_leaves() + i;
            println!("set_idx: {}", set_idx);
            tree.set(set_idx, value);
            println!("tree: {}", tree);
        }

        let expected_root =
            hex!("57054e43fa56333fd51343b09460d48b9204999c376624f52480c5593b91eff4");
        println!("root: {:x?}, expected: {:x?}", tree.root(), expected_root);
        assert_eq!(tree.root(), expected_root);

        let proof = tree.proof(3);
        assert!(proof.path.len() == 4);
        let expected_proof = MerkleProof {
            path: vec![
                (Direction::Right, [0x22; 32]),
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

    #[test]
    fn test_complex_merkle_tree2() {
        let initial_leaf = [0x00; 32];
        let mut tree = MerkleTree::new(5, initial_leaf);

        println!("tree: {}", tree);
        //for i in ((tree.num_leaves())..(2 * tree.num_leaves() - 1)).rev() {
        for i in 0..tree.num_leaves() {
            let mut value: [u8; 32] = [0; 32];
            // let asdf = M256::from_bytes_be(value);
            let val_int: M256 = M256::from(i as u64)
                * M256::from_str(
                    "1111111111111111111111111111111111111111111111111111111111111111",
                )
                .unwrap();
            val_int.0.to_big_endian(&mut value);
            println!("value: {:x?}", value);
            let set_idx = tree.num_nodes() - tree.num_leaves() + i;
            println!("set_idx: {}", set_idx);
            tree.set(set_idx, value);
            println!("tree: {}", tree);
        }

        let mut leaf_5: [u8; 32] = [0; 32];
        let leaf_5_int: M256 = M256::from(5 as u64)
            * M256::from_str("1111111111111111111111111111111111111111111111111111111111111111")
                .unwrap();
        leaf_5_int.0.to_big_endian(&mut leaf_5);

        let root = tree.root();
        // In the example given in the problem statement this was proof(3), but I am quite
        // sure that is wrong and it should be proof(5), which makes sense aesthetically,
        // intuitively, and it gives the right answer.
        let proof = tree.proof(5);
        assert!(proof.path.len() == 4);
        assert_eq!(verify(&proof, leaf_5), root);
    }
}
