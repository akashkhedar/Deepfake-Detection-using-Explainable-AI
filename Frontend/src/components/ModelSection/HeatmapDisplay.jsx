import { Analytics } from "@mui/icons-material";
import {
  Box,
  Chip,
  Paper,
  Typography,
  useMediaQuery,
  useTheme,
} from "@mui/material";
import PropTypes from "prop-types";

import { CheckCircle, Warning } from "@mui/icons-material";

const HeatmapDisplay = ({
  heatmapUrl,
  prediction,
  isAnalyzing,
  confidence,
}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("md"));

  return (
    <Box sx={{ textAlign: "center" }}>
      <Paper
        sx={{
          position: "relative",
          width: isMobile ? "100%" : "25rem",
          height: isMobile ? "20rem" : "25rem", // Smaller height on mobile
          borderRadius: 3,
          overflow: "hidden",
          border: "2px solid",
          borderColor: "primary.main",
          mb: 1,
          bgcolor: "black",
          flexShrink: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {heatmapUrl ? (
          <Box
            component="img"
            src={heatmapUrl}
            alt="Grad-CAM Heatmap"
            sx={{
              width: "100%",
              height: "100%",
              objectFit: "contain",
              display: "block",
            }}
          />
        ) : isAnalyzing ? (
          <Box sx={{ textAlign: "center", color: "white" }}>
            <Analytics
              sx={{
                fontSize: isMobile ? 40 : 48,
                color: "primary.main",
                mb: 1,
                animation: "pulse 2s infinite",
              }}
            />
            <Typography
              variant={isMobile ? "caption" : "body2"}
              color="primary.main"
            >
              Generating heatmap...
            </Typography>
          </Box>
        ) : (
          <Box sx={{ textAlign: "center", color: "white" }}>
            <Analytics
              sx={{
                fontSize: isMobile ? 40 : 48,
                color: "text.disabled",
                mb: 1,
              }}
            />
            <Typography
              variant={isMobile ? "caption" : "body2"}
              color="text.secondary"
            >
              Heatmap will appear here
            </Typography>
          </Box>
        )}
      </Paper>

      {/* Prediction Status */}
      {(prediction || isAnalyzing) && (
        <Box
          sx={{
            mt: 1.9,
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            gap: 1,
            flexWrap: "wrap",
          }}
        >
          {isAnalyzing ? (
            <Chip
              label="Analyzing..."
              color="info"
              size={isMobile ? "small" : "medium"}
              sx={{ fontSize: isMobile ? "0.75rem" : "0.875rem", px: 2 }}
            />
          ) : (
            <>
              <Chip
                icon={prediction === "real" ? <CheckCircle /> : <Warning />}
                label={
                  prediction === "real" ? "Real Image" : "Fake Image Detected"
                }
                color={prediction === "real" ? "success" : "error"}
                size={isMobile ? "small" : "medium"}
                sx={{ fontSize: isMobile ? "0.75rem" : "0.875rem", px: 2 }}
              />
              {typeof confidence === "number" && (
                <Chip
                  label={`Confidence: ${(confidence * 100).toFixed(1)}%`}
                  variant="outlined"
                  size={isMobile ? "small" : "medium"}
                  sx={{ fontSize: isMobile ? "0.75rem" : "0.875rem", px: 2 }}
                />
              )}
            </>
          )}
        </Box>
      )}
    </Box>
  );
};

HeatmapDisplay.propTypes = {
  heatmapUrl: PropTypes.string,
  prediction: PropTypes.oneOf(["real", "fake", null]),
  isAnalyzing: PropTypes.bool.isRequired,
  confidence: PropTypes.number,
};

export default HeatmapDisplay;
