import { useState, useCallback } from "react";

const useImageAnalysis = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [heatmapUrl, setHeatmapUrl] = useState(null);
  const [explanation, setExplanation] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const analyzeImage = useCallback(async (imageFile) => {
    setIsAnalyzing(true);
    setPrediction(null);
    setHeatmapUrl(null);
    setExplanation("");

    try {
      const formData = new FormData();
      formData.append("file", imageFile);
      console.log("Sending request to backend...", formData);

      const response = await fetch("http://localhost:8000/predict/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      const predictionLower = result.prediction.toLowerCase();
      setPrediction(predictionLower);

      // Set the heatmap from the base64 string returned by the API
      if (result.heatmap) {
        setHeatmapUrl(`data:image/png;base64,${result.heatmap}`);
      }

      // Set explanation only for fake images
      if (predictionLower === "fake") {
        setExplanation(
          "The model detected artificial patterns commonly found in AI-generated images. Key indicators include unusual texture inconsistencies, unnatural lighting gradients, and synthetic-looking features. The attention heatmap shows areas where the model detected digital artifacts and inconsistent patterns typical of deepfake generation."
        );
      }
    } catch (error) {
      console.error("Error analyzing image:", error);
      setPrediction("error");
      setExplanation(
        "Failed to analyze the image. Please make sure the backend server is running on 127.0.0.1:8000 and try again."
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
          analyzeImage(file);
        };
        reader.readAsDataURL(file);
      }
    };
    input.click();
  }, [analyzeImage]);

  const handleFileSelect = useCallback(
    (file) => {
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          setSelectedImage(e.target.result);
          analyzeImage(file);
        };
        reader.readAsDataURL(file);
      }
    },
    [analyzeImage]
  );

  const handleReset = useCallback(() => {
    setSelectedImage(null);
    setPrediction(null);
    setHeatmapUrl(null);
    setExplanation("");
    setIsAnalyzing(false);
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
  };
};

export default useImageAnalysis;
