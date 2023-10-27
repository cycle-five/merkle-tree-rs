// use std::fmt::{Display, Formatter};

use sha3::{digest::core_api::CoreWrapper, Digest, Sha3_256, Sha3_256Core};
/// Represents a Merkle Tree with methods for constructing and verifying proofs.
pub struct MerkleTree {
    nodes: Vec<[u8; 32]>,
    depth: u32,
    num_leaves: u32,
    num_nodes: u32,
}

#[derive(Debug, PartialEq)]
enum Direction {
    Left,
    Right,
}

#[derive(Debug, PartialEq)]
pub struct MerkleProof {
    path: Vec<(Direction, [u8; 32])>,
}

impl MerkleTree {
    /// Creates a new MerkleTree with the specified depth and initial leaf value.
    pub fn new(depth: u32, initial_leaf: [u8; 32]) -> Self {
        let num_leaves = 2u32.pow(depth);
        let num_nodes = 2 * num_leaves - 1;

        // It was a bit ambigious in the problem definition but it seems the
        // leaves are indeed hashed values.
        let initial_leaf = Self::compute_hash_single(initial_leaf);
        let mut nodes = vec![initial_leaf; num_nodes as usize];

        for i in (num_leaves - 1)..(2 * num_leaves - 1) {
            nodes[i as usize] = initial_leaf;
        }

        for i in (0..(num_leaves - 2)).rev() {
            let left_child = left_child_index(i);
            let right_child = left_child + 1;

            nodes[i as usize] =
                Self::compute_hash(nodes[left_child as usize], nodes[right_child as usize]);
        }

        MerkleTree {
            nodes,
            depth,
            num_leaves,
            num_nodes,
        }
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

        self.nodes[leaf_index as usize] = Self::compute_hash_single(value);
        let mut current_index = leaf_index;

        while let Some(parent) = parent_index(current_index) {
            let left_child = left_child_index(parent);
            let right_child = left_child + 1;
            println!(
                "parent: {}, left_child: {}, right_child: {}",
                parent, left_child, right_child
            );

            self.nodes[parent as usize] = Self::compute_hash(
                self.nodes[left_child as usize],
                self.nodes[right_child as usize],
            );
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
                (current_index + 1, Direction::Right)
            } else {
                // path.push((Direction::Left, self.nodes[current_index as usize - 1]));
                (current_index - 1, Direction::Left)
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

// impl Display for MerkleTree {
//     fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
//         let mut s = String::new();
//         let mut current_index = 0;
//         let mut current_depth = 0;
//         let mut current_offset = 0;

//         while current_index < self.num_nodes {
//             let (depth, offset) = index_to_depth_offset(current_index as usize);
//             if depth != current_depth {
//                 current_depth = depth;
//                 current_offset = 0;
//                 s.push('\n');
//             }

//             let num_spaces = depth;
//             for _ in 0..num_spaces {
//                 s.push(' ');
//             }

//             s.push_str(&format!("{:x?}", self.nodes[current_index as usize]));

//             current_index += 1;
//             current_offset += 1;
//         }

//         write!(f, "{}", s)
//     }
// }

pub fn depth_offset_to_index(depth: u32, offset: u32) -> u32 {
    (2u32.pow(depth) - 1) + offset
}

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

fn parent_index(index: u32) -> Option<u32> {
    if index == 0 {
        println!("index is zero");
        None
    } else {
        Some((index - 1) / 2)
    }
}

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
mod tests {
    use super::*;
    use hex::FromHex;
    use hex_literal::hex;

    #[test]
    fn test_asdf() {
        for i in 0..10 {
            println!("{}", i);
        }

        for i in (0..10).rev() {
            println!("{}", i);
        }
    }

    #[test]
    fn test_hex_decoding_encoding() {
        // let hex = "d4490f4d374ca8a44685fe9471c5b8dbe58cdffd13d30d9aba15dd29efb92930";
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
    fn test_merkle_tree_root() {
        let initial_leaf = [0xab; 32];
        let tree = MerkleTree::new(20, initial_leaf);
        let expected_root = <[u8; 32]>::from_hex(
            "d4490f4d374ca8a44685fe9471c5b8dbe58cdffd13d30d9aba15dd29efb92930",
        )
        .unwrap();
        assert_eq!(tree.root(), expected_root);
    }

    #[test]
    fn test_merkle_tree_set() {
        let initial_leaf = [0xab; 32];
        let zero_leaf = [0x00; 32];
        let expected_root = <[u8; 32]>::from_hex(
            "d4490f4d374ca8a44685fe9471c5b8dbe58cdffd13d30d9aba15dd29efb92930",
            // "57054e43fa56333fd51343b09460d48b9204999c376624f52480c5593b91eff4",
        )
        .unwrap();
        let mut tree = MerkleTree::new(5, initial_leaf);
        // println!("tree: {}", tree);
        tree.set(2u32.pow(5) - 1, zero_leaf);
        // println!("tree: {}", tree);
        assert_ne!(tree.root(), expected_root);
        tree.set(2u32.pow(5) - 1, initial_leaf);
        // println!("tree: {}", tree);
        assert_eq!(tree.root(), expected_root);
    }

    // #[test]
    // fn test_merkle_tree_proof() {
    //     let initial_leaf = [0x00; 32];
    //     let tree = MerkleTree::new(5, initial_leaf);
    //     for i in 0..tree.num_leaves() {
    //         let proof = tree.proof(i);
    //         assert_eq!(verify(&proof, initial_leaf), tree.root());
    //     }
    // }

    #[test]
    fn test_merkle_tree_set_and_proof() {
        let initial_leaf = [0x00; 32];
        let mut tree = MerkleTree::new(3, initial_leaf);
        tree.set(0, [0x11; 32]);
        let proof = tree.proof(0);
        assert_eq!(verify(&proof, [0x11; 32]), tree.root());
    }

    use bigint::M256;
    use std::str::FromStr;
    // use etcommon_bigint::M256;
    #[test]
    fn test_complex_merkle_tree() {
        let initial_leaf = [0x00; 32];
        let mut tree = MerkleTree::new(5, initial_leaf);

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