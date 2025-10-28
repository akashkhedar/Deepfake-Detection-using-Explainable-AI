import { useState, useCallback, useEffect } from "react";

// In Vite projects, env vars are available under import.meta.env
const API_BASE =
  (typeof import.meta !== "undefined" &&
    import.meta.env &&
    import.meta.env.VITE_API_BASE) ||
  "http://localhost:8000";

const useImageAnalysis = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [heatmapUrl, setHeatmapUrl] = useState(null);
  const [explanation, setExplanation] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null); // null => ensemble
  const [confidence, setConfidence] = useState(null);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/models/`);
        if (!res.ok) return;
        const j = await res.json();
        if (mounted && j.available_models) {
          setAvailableModels(j.available_models);
        }
      } catch (e) {
        console.log(e);
        // ignore
      }
    })();
    return () => (mounted = false);
  }, []);

  const analyzeImage = useCallback(async (imageFile, model = null) => {
    setIsAnalyzing(true);
    setPrediction(null);
    setHeatmapUrl(null);
    setExplanation("");
    setConfidence(null);

    try {
      const formData = new FormData();
      formData.append("file", imageFile);
      const url = model
        ? `${API_BASE}/predict/?model=${encodeURIComponent(model)}`
        : `${API_BASE}/predict/`;
      const response = await fetch(url, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      // backend returns either single-model response with 'prediction' or ensemble response with 'ensemble_prediction'
      const pred = result.prediction || result.ensemble_prediction;
      const predictionLower = pred ? pred.toLowerCase() : "error";
      setPrediction(predictionLower);

      // set overall confidence (single-model or ensemble)
      if (typeof result.confidence === "number") {
        setConfidence(result.confidence);
      } else if (typeof result.ensemble_confidence === "number") {
        setConfidence(result.ensemble_confidence);
      }

      if (result.heatmap) {
        setHeatmapUrl(`data:image/png;base64,${result.heatmap}`);
      }

      if (result.explanation) {
        setExplanation(result.explanation);
      } else {
        setExplanation(
          predictionLower === "fake"
            ? "The model detected artificial patterns commonly found in AI-generated images."
            : "The model detected patterns consistent with authentic images."
        );
      }
    } catch (error) {
      console.error("Error analyzing image:", error);
      setPrediction("error");
      setExplanation(
        `Failed to analyze the image. Please make sure the backend server is running at ${API_BASE} and try again.`
      );
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  const handleImageUpload = useCallback(() => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.onchange = (e) => {
      const file = e.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          setSelectedImage(e.target.result);
          analyzeImage(file, selectedModel);
        };
        reader.readAsDataURL(file);
      }
    };
    input.click();
  }, [analyzeImage, selectedModel]);

  const handleFileSelect = useCallback(
    (file) => {
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          setSelectedImage(e.target.result);
          analyzeImage(file, selectedModel);
        };
        reader.readAsDataURL(file);
      }
    },
    [analyzeImage, selectedModel]
  );

  const handleReset = useCallback(() => {
    setSelectedImage(null);
    setPrediction(null);
    setHeatmapUrl(null);
    setExplanation("");
    setIsAnalyzing(false);
    setConfidence(null);
  }, []);

  return {
    selectedImage,
    prediction,
    heatmapUrl,
    explanation,
    isAnalyzing,
    handleImageUpload,
    handleFileSelect,
    handleReset,
    analyzeImage,
    availableModels,
    selectedModel,
    setSelectedModel,
    confidence,
  };
};

export default useImageAnalysis;
