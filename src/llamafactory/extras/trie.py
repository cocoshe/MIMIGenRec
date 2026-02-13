# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional


class Trie:
    """Trie for constrained generation to restrict next token candidates."""

    def __init__(self, sequences: Optional[list[list[int]]] = None):
        self.trie_dict: dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie: Optional[Trie] = None
        self.bos_token_id: Optional[int] = None

    def append(self, trie: "Trie", bos_token_id: int) -> None:
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: list[int]) -> None:
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: list[int]) -> list[int]:
        return Trie._get_from_trie(prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id)

    @staticmethod
    def load_from_dict(trie_dict: dict) -> "Trie":
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: list[int], trie_dict: dict) -> None:
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: list[int],
        trie_dict: dict,
        append_trie: Optional["Trie"] = None,
        bos_token_id: Optional[int] = None,
    ) -> list[int]:
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id is not None and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence: list[int], trie_dict: dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(prefix_sequence + [next_token], trie_dict[next_token])
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, value: list[int]) -> list[int]:
        return self.get(value)
