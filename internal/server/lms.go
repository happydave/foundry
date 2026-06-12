package server

// LM Studio native API response types for GET /api/v1/models.

type lmsModelsResponse struct {
	Models []lmsModelEntry `json:"models"`
}

type lmsModelEntry struct {
	Key             string              `json:"key"`
	Type            string              `json:"type"`
	Publisher       string              `json:"publisher"`
	DisplayName     string              `json:"display_name"`
	Architecture    string              `json:"architecture"`
	SizeBytes       int64               `json:"size_bytes"`
	ContextLength   uint32              `json:"context_length"`
	Quantization    lmsQuantization     `json:"quantization"`
	LoadedInstances []lmsLoadedInstance `json:"loaded_instances"`
}

type lmsQuantization struct {
	Name          string  `json:"name"`
	BitsPerWeight float64 `json:"bits_per_weight"`
}

type lmsLoadedInstance struct {
	ID     string            `json:"id"`
	Config lmsInstanceConfig `json:"config"`
}

type lmsInstanceConfig struct {
	ContextLength  int  `json:"context_length"`
	EvalBatchSize  int  `json:"eval_batch_size"`
	FlashAttention bool `json:"flash_attention"`
	Parallel       int  `json:"parallel"`
}

// bitsPerWeightTable maps common GGUF quantization names to approximate bits-per-weight.
// Returns 0.0 for unrecognised names.
var bitsPerWeightTable = map[string]float64{
	// Standard K-quants
	"Q2_K":   2.63,
	"Q3_K_S": 3.50,
	"Q3_K_M": 3.91,
	"Q3_K_L": 4.27,
	"Q4_0":   4.55,
	"Q4_K_S": 4.37,
	"Q4_K_M": 4.50,
	"Q5_0":   5.00,
	"Q5_K_S": 5.21,
	"Q5_K_M": 5.33,
	"Q6_K":   6.14,
	"Q8_0":   8.00,
	// Float types
	"F16":  16.00,
	"BF16": 16.00,
	"F32":  32.00,
	// IQ (importance-matrix) quants
	"IQ1_S":   1.56,
	"IQ1_M":   1.75,
	"IQ2_XXS": 2.06,
	"IQ2_XS":  2.31,
	"IQ2_S":   2.50,
	"IQ2_M":   2.70,
	"IQ3_XXS": 3.06,
	"IQ3_XS":  3.30,
	"IQ3_S":   3.50,
	"IQ3_M":   3.70,
	"IQ4_XS":  4.25,
	"IQ4_NL":  4.50,
}

func bitsPerWeight(quantization string) float64 {
	return bitsPerWeightTable[quantization]
}
