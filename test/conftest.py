import pytest
import torch
import contextlib # Used in grpo_core, ensure available if tests trigger it

# ===============================================================================
# DUMMY CONFIG FOR MODEL
# ===============================================================================
class DummyModelConfig:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.vocab_size = 32000 # Example vocab size
        # Add any other config attributes your main code might access from model.config
        # e.g., model.config.is_encoder_decoder, model.config.num_hidden_layers, etc.
        # For GRPO, pad_token_id and eos_token_id are the most critical.

# ===============================================================================
# DUMMY TOKENIZER
# ===============================================================================
class DummyTokenizer:
    def __init__(self, model_max_length=512):
        self.pad_token_id = 0
        self.pad_token = '[PAD]'
        self.eos_token_id = 1
        self.eos_token = '[EOS]'
        self.unk_token_id = 2
        self.unk_token = '[UNK]'
        self.model_max_length = model_max_length
        
        # Simple vocab for testing decoding.
        self.vocab = {
            self.pad_token: self.pad_token_id,
            self.eos_token: self.eos_token_id,
            self.unk_token: self.unk_token_id,
            "hello": 10, "world": 11, "what": 12, "is": 13, "2+2": 14,
            "a": 15, "b": 16, "c": 17, "d": 18, "=": 19, "+": 20, "foo":21, "bar":22,
        }
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

    def add_special_tokens(self, special_tokens_dict):
        added_tokens = 0
        if 'pad_token' in special_tokens_dict:
            self.pad_token = special_tokens_dict['pad_token']
            # self.pad_token_id might need update if vocab uses it
            if self.pad_token not in self.vocab:
                # simplistic add
                new_id = max(self.vocab.values()) + 1
                self.vocab[self.pad_token] = new_id
                self.ids_to_tokens[new_id] = self.pad_token
                self.pad_token_id = new_id # Update pad_token_id
                added_tokens +=1
        if 'eos_token' in special_tokens_dict:
            self.eos_token = special_tokens_dict['eos_token']
            if self.eos_token not in self.vocab:
                new_id = max(self.vocab.values()) + 1
                self.vocab[self.eos_token] = new_id
                self.ids_to_tokens[new_id] = self.eos_token
                self.eos_token_id = new_id
                added_tokens +=1
        # Handle other special tokens if necessary
        return added_tokens

    def batch_decode(self, sequences, skip_special_tokens=False, **kwargs):
        decoded_texts = []
        for seq in sequences:
            if isinstance(seq, torch.Tensor):
                seq = seq.tolist()
            current_text = []
            for token_id in seq:
                if skip_special_tokens and token_id in [self.pad_token_id, self.eos_token_id]:
                    continue
                current_text.append(self.ids_to_tokens.get(token_id, self.unk_token))
            decoded_texts.append(" ".join(current_text))
        return decoded_texts

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        return self.batch_decode([token_ids], skip_special_tokens=skip_special_tokens, **kwargs)[0]

    def __call__(self, text_inputs, return_tensors="pt", padding=True, truncation=True, max_length=None, **kwargs):
        if isinstance(text_inputs, str):
            text_inputs = [text_inputs]

        effective_max_length = max_length if max_length is not None else self.model_max_length

        all_input_ids = []
        for text in text_inputs:
            # Super simple tokenization: split by space, map to vocab, or use fixed length
            # For tests, let's use a small, somewhat variable prompt length based on split.
            tokens = text.split()
            token_ids = [self.vocab.get(t, self.unk_token_id) for t in tokens]
            
            if truncation and len(token_ids) > effective_max_length:
                token_ids = token_ids[:effective_max_length]
            
            all_input_ids.append(token_ids)

        # Determine max length in batch for padding
        current_max_len_in_batch = 0
        if not all_input_ids: # Handle empty input list
             current_max_len_in_batch = 0
        elif isinstance(all_input_ids[0], torch.Tensor): # If already tensors
            current_max_len_in_batch = max(seq.size(0) for seq in all_input_ids) if all_input_ids else 0
        else: # If lists of ids
            current_max_len_in_batch = max(len(seq) for seq in all_input_ids) if all_input_ids else 0


        if padding == "max_length" or (padding and max_length is not None):
            pad_to_length = effective_max_length
        elif padding: # Pad to the longest sequence in the batch
            pad_to_length = current_max_len_in_batch
        else: # No padding
            pad_to_length = 0 # Each sequence will have its own length (can cause issues for stacking)


        final_input_ids_list = []
        final_attention_mask_list = []

        for token_ids in all_input_ids:
            seq_len = len(token_ids)
            if padding and seq_len < pad_to_length:
                padding_needed = pad_to_length - seq_len
                padded_ids = token_ids + [self.pad_token_id] * padding_needed
                attention_mask = [1] * seq_len + [0] * padding_needed
            else: # No padding or sequence is already at/beyond pad_to_length (if truncation happened)
                padded_ids = token_ids
                attention_mask = [1] * seq_len
            
            final_input_ids_list.append(torch.tensor(padded_ids, dtype=torch.long))
            final_attention_mask_list.append(torch.tensor(attention_mask, dtype=torch.long))
        
        if not final_input_ids_list: # if text_inputs was empty
             if return_tensors == "pt":
                return { # Return empty tensors matching expected structure
                    "input_ids": torch.empty((0,0), dtype=torch.long),
                    "attention_mask": torch.empty((0,0), dtype=torch.long)
                }

        if return_tensors == "pt":
            # This assumes all tensors in the list are now of the same length due to padding.
            # If padding=False and lengths vary, stack will fail.
            # This is a common simplification for dummy tokenizers.
            try:
                input_ids_tensor = torch.stack(final_input_ids_list)
                attention_mask_tensor = torch.stack(final_attention_mask_list)
            except RuntimeError as e:
                # Handle cases where sequences might not be same length if padding=False
                if "stack expects each tensor to be equal size" in str(e) and not padding:
                    # This is expected if padding=False and sequences have variable lengths.
                    # The caller (e.g., model.generate) might handle this or error.
                    # For this dummy, we'll let it error or return unstacked if that's useful.
                    # Or, we could force padding to longest if padding is not explicitly False.
                    # Let's print a warning and return un-padded for this specific case
                    # print("Warning: DummyTokenizer returning unstacked tensors due to varying lengths and padding=False.")
                    # For most tests, padding=True will be used.
                    pass # Let it raise for now if problem for specific test
                raise e


            class BatchEncodingMock(dict):
                def __init__(self, data):
                    super().__init__(data)
                    self.data = data 
                def to(self, device_str_or_obj):
                    # device = torch.device(device_str_or_obj) # Not needed if already on device
                    for k, v in self.items(): # Iterate over self.items() which is the dict
                        if isinstance(v, torch.Tensor):
                            self[k] = v.to(device_str_or_obj)
                    return self
            
            return BatchEncodingMock({
                "input_ids": input_ids_tensor,
                "attention_mask": attention_mask_tensor
            })
        
        raise NotImplementedError("DummyTokenizer currently only supports return_tensors='pt'")

# ===============================================================================
# DUMMY MODEL
# ===============================================================================
class DummyModel:
    def __init__(self, config=None):
        self.config = config if config else DummyModelConfig()
        self._actual_device = torch.device("cpu") 
        self.dummy_param = torch.nn.Parameter(torch.randn(1, 1)) # For optimizer

    def resize_token_embeddings(self, new_num_tokens):
        # In a real model, this would resize the embedding layer.
        # For a dummy, we can update the config's vocab_size if it's used.
        # self.config.vocab_size = new_num_tokens
        # Or reinitialize a dummy embedding layer if we had one.
        return None 

    def to(self, device_str_or_obj):
        self._actual_device = torch.device(device_str_or_obj)
        self.dummy_param = self.dummy_param.to(self._actual_device)
        return self

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self

    def generate(self, input_ids, attention_mask=None, max_new_tokens=5, pad_token_id=None, eos_token_id=None, **kwargs):
        batch_size, prompt_len = input_ids.shape
        
        # Use model's configured eos_token_id if not provided, otherwise fallback for generation
        effective_eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        if effective_eos_token_id is None: effective_eos_token_id = 1 # Default if still none

        # Dummy generated tokens: just repeat a token or use eos
        gen_token_values = [effective_eos_token_id] * max_new_tokens # Simplistic: always generate EOS
        new_tokens_tensor = torch.tensor([gen_token_values] * batch_size, dtype=torch.long, device=input_ids.device)
        
        return torch.cat([input_ids, new_tokens_tensor], dim=1)

    def __call__(self, input_ids, attention_mask=None, labels=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        vocab_size = self.config.vocab_size
        
        # Ensure logits are on the same device as the model's dummy_param (and thus input_ids if to() was called)
        dummy_logits = torch.randn(batch_size, seq_len, vocab_size, 
                                   device=self.dummy_param.device, dtype=torch.float32)
        dummy_logits.requires_grad_(True) # Ensure logits require grad for loss computation tests

        class DummyOutput:
            def __init__(self, logits_tensor):
                self.logits = logits_tensor
                self.loss = None
                # HuggingFace CausalLMOutput often includes 'past_key_values', etc.
                # Add them as None if your code under test might try to access them.
                self.past_key_values = None

            def to(self, device_str_or_obj): # Usually not called on output object itself
                self.logits = self.logits.to(torch.device(device_str_or_obj))
                if self.loss is not None: self.loss = self.loss.to(torch.device(device_str_or_obj))
                return self

        output = DummyOutput(dummy_logits)

        if labels is not None:
            # Simplified loss: just a sum of logits for target labels (not real CE) or random
            # This ensures 'loss' is a tensor that requires_grad if logits do.
            # A more realistic dummy loss for testing backprop:
            # target_logits = dummy_logits.view(-1, vocab_size)
            # target_labels = labels.view(-1)
            # dummy_loss = torch.nn.functional.cross_entropy(
            #     target_logits[:target_labels.size(0)], # Ensure correct slicing if shapes differ
            #     target_labels,
            #     ignore_index=self.config.pad_token_id # if applicable
            # )
            # Simpler:
            dummy_loss = (dummy_logits * 0.001).mean() # Ensure it depends on logits
            if dummy_loss.grad_fn is None: # if it somehow became a leaf
                 dummy_loss.requires_grad_(True)
            output.loss = dummy_loss
            
        return output

    def parameters(self):
        return [self.dummy_param]

    def get_input_embeddings(self): # For resize_token_embeddings if it internally uses it
        class DummyEmbedding:
            def __init__(self, weight):
                self.weight = weight
        return DummyEmbedding(torch.nn.Parameter(torch.randn(self.config.vocab_size, 10))) # (vocab_size, hidden_size)


# ===============================================================================
# FAKE AUTO CLASSES (WRAPPERS FOR DUMMIES)
# ===============================================================================
class FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(model_name_or_path, **kwargs):
        # print(f"CONFTESLog: FakeAutoTokenizer: Loading dummy for {model_name_or_path}")
        return DummyTokenizer(**kwargs) # Pass kwargs like model_max_length

class FakeAutoModel:
    @staticmethod
    def from_pretrained(model_name_or_path, **kwargs):
        # print(f"CONFTESLog: FakeAutoModel: Loading dummy for {model_name_or_path}")
        # Config can be passed or a default DummyModelConfig used.
        # If torch_dtype is in kwargs, DummyModel needs to handle it (e.g. set self.dtype)
        # For now, DummyModel uses float32 for logits and params are created as default.
        config_kwargs = {}
        if 'torch_dtype' in kwargs: # Store it on config if model uses it
            # config_kwargs['torch_dtype'] = kwargs['torch_dtype']
            pass # DummyModel doesn't strictly use dtype for now, params are float32
        
        return DummyModel(config=DummyModelConfig())