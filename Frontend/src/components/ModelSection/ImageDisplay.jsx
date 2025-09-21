import React from "react";
import {
  Box,
  Typography,
  Button,
  Chip,
  useTheme,
  useMediaQuery,
} from "@mui/material";
import { CheckCircle, Warning } from "@mui/icons-material";
import PropTypes from "prop-types";

const ImageDisplay = ({ image, prediction, isAnalyzing, onReset }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("md"));

  return (
    <Box>
      <Box
        sx={{
          position: "relative",
          width: "100%",
          aspectRatio: "1 / 1", // Force square
          borderRadius: 3,
          overflow: "hidden",
          border: "2px solid",
          borderColor: "primary.main",
          mb: 2,
        }}
      >
        <Box
          component="img"
          src={image}
          alt="Selected for analysis"
          sx={{
            width: "100%",
            height: "100%",
            objectFit: "contain",
            display: "block",
            backgroundColor: "black",
          }}
        />
      </Box>

      {/* Prediction Status */}
      {(prediction || isAnalyzing) && (
        <Box sx={{ textAlign: "center", mb: 2 }}>
          {isAnalyzing ? (
            <Chip
              label="Analyzing..."
              color="info"
              size={isMobile ? "medium" : "large"}
              sx={{ fontSize: isMobile ? "0.875rem" : "1rem", p: 1 }}
            />
          ) : (
            <Chip
              icon={prediction === "real" ? <CheckCircle /> : <Warning />}
              label={
                prediction === "real" ? "Real Image" : "Fake Image Detected"
              }
              color={prediction === "real" ? "success" : "error"}
              size={isMobile ? "medium" : "large"}
              sx={{ fontSize: isMobile ? "0.875rem" : "1rem", p: 1 }}
            />
          )}
        </Box>
      )}

      <Button
        variant="outlined"
        fullWidth
        onClick={onReset}
        sx={{ mt: 1 }}
        disabled={isAnalyzing}
      >
        Upload Different Image
      </Button>
    </Box>
  );
};

ImageDisplay.propTypes = {
  image: PropTypes.string.isRequired,
  prediction: PropTypes.oneOf(["real", "fake", null]),
  isAnalyzing: PropTypes.bool.isRequired,
  onReset: PropTypes.func.isRequired,
};

export default ImageDisplay;
