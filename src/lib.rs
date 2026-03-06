/*
 * Copyright (c) 2024 Works Applications Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use std::sync::Arc;

use serde::Serialize;
use wasm_bindgen::prelude::*;

use sudachi::analysis::stateless_tokenizer::StatelessTokenizer;
use sudachi::analysis::Tokenize;
use sudachi::dic::dictionary::JapaneseDictionary;
use sudachi::prelude::*;

/// Install a panic hook that forwards Rust panics to `console.error` in JS.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Split granularity for tokenization.
///
/// - `"A"` – shortest units
/// - `"B"` – middle units
/// - `"C"` – named-entity / longest units (default)
fn parse_mode(s: &str) -> Mode {
    match s.to_uppercase().as_str() {
        "A" => Mode::A,
        "B" => Mode::B,
        _ => Mode::C,
    }
}

/// A single morpheme returned from [`Tokenizer::tokenize`].
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct MorphemeData {
    /// Surface form as it appears in the input text.
    surface: String,
    /// Dictionary (base) form.
    dictionary_form: String,
    /// Reading in katakana.
    reading_form: String,
    /// Normalized form.
    normalized_form: String,
    /// Part-of-speech components (up to 6 fields, e.g. `["名詞","普通名詞","一般","*","*","*"]`).
    part_of_speech: Vec<String>,
    /// `true` when the word was not found in any dictionary (out-of-vocabulary).
    is_oov: bool,
    /// Start character offset in the original string (Unicode code-point index).
    begin: usize,
    /// End character offset in the original string (Unicode code-point index, exclusive).
    end: usize,
}

/// A Sudachi tokenizer that holds a loaded dictionary.
///
/// Construct with `new Tokenizer(dictBytes)` where `dictBytes` is a `Uint8Array`
/// containing the raw bytes of a Sudachi system dictionary (`.dic` file).
#[wasm_bindgen]
pub struct Tokenizer {
    inner: StatelessTokenizer<Arc<JapaneseDictionary>>,
}

#[wasm_bindgen]
impl Tokenizer {
    /// Create a tokenizer from the raw bytes of a Sudachi system dictionary.
    ///
    /// ```js
    /// const response = await fetch("system_core.dic");
    /// const dictBytes = new Uint8Array(await response.arrayBuffer());
    /// const tokenizer = new Tokenizer(dictBytes);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(dict_bytes: &[u8]) -> Result<Tokenizer, JsError> {
        let dict = Arc::new(
            JapaneseDictionary::from_system_bytes(dict_bytes.to_vec())
                .map_err(|e| JsError::new(&e.to_string()))?,
        );
        Ok(Tokenizer {
            inner: StatelessTokenizer::new(dict),
        })
    }

    /// Tokenize Japanese text and return an array of morpheme objects.
    ///
    /// @param text  - Input Japanese text.
    /// @param mode  - Split mode: `"A"` (short), `"B"` (middle), or `"C"` (default, named-entity).
    /// @returns     Array of `{ surface, dictionaryForm, readingForm, normalizedForm,
    ///              partOfSpeech, isOov, begin, end }` objects.
    ///
    /// ```js
    /// const morphemes = tokenizer.tokenize("今日はいい天気ですね。", "C");
    /// for (const m of morphemes) {
    ///   console.log(m.surface, m.readingForm, m.partOfSpeech.join("-"));
    /// }
    /// ```
    pub fn tokenize(&self, text: &str, mode: &str) -> Result<JsValue, JsError> {
        let result = self
            .inner
            .tokenize(text, parse_mode(mode), false)
            .map_err(|e| JsError::new(&e.to_string()))?;

        let morphemes: Vec<MorphemeData> = result
            .iter()
            .map(|m| MorphemeData {
                surface: m.surface().to_string(),
                dictionary_form: m.dictionary_form().to_string(),
                reading_form: m.reading_form().to_string(),
                normalized_form: m.normalized_form().to_string(),
                part_of_speech: m.part_of_speech().to_vec(),
                is_oov: m.is_oov(),
                begin: m.begin_c(),
                end: m.end_c(),
            })
            .collect();

        serde_wasm_bindgen::to_value(&morphemes).map_err(|e| JsError::new(&e.to_string()))
    }
}
