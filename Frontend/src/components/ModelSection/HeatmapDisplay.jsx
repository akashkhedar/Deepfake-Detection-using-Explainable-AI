import React from "react";
import { Box, Typography, Paper, useTheme, useMediaQuery } from "@mui/material";
import { Analytics } from "@mui/icons-material";
import PropTypes from "prop-types";

const HeatmapDisplay = ({ heatmapUrl, isAnalyzing }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("md"));

  return (
    <Paper
      sx={{
        position: "relative",
        width: "100%",
        aspectRatio: "1 / 1", // Same square ratio as image
        borderRadius: 3,
        overflow: "hidden",
        border: "2px solid",
        borderColor: "secondary.main",
        mb: 2,
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
            backgroundColor: "black",
          }}
        />
      ) : isAnalyzing ? (
        <Box sx={{ textAlign: "center" }}>
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
        <Box sx={{ textAlign: "center" }}>
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
  );
};

HeatmapDisplay.propTypes = {
  heatmapUrl: PropTypes.string,
  isAnalyzing: PropTypes.bool.isRequired,
};

export default HeatmapDisplay;
