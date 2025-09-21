import React from "react";
import { Box, Typography, Paper } from "@mui/material";
import { Psychology, Error, CheckCircle } from "@mui/icons-material";
import { useMediaQuery, useTheme } from "@mui/material";
import PropTypes from "prop-types";

const ExplanationBox = ({ explanation, isAnalyzing, prediction }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("md"));
  const isError = prediction === "error";
  const isReal = prediction === "real";

  // Determine styling based on prediction type
  const getBoxStyling = () => {
    if (isError) {
      return {
        bgcolor: "warning.dark",
        borderColor: "warning.main",
        iconColor: "warning.main",
        textColor: "warning.main",
      };
    } else if (isReal) {
      return {
        bgcolor: "success.dark",
        borderColor: "success.main",
        iconColor: "success.main",
        textColor: "success.main",
      };
    } else {
      return {
        bgcolor: "error.dark",
        borderColor: "error.main",
        iconColor: "error.main",
        textColor: "error.main",
      };
    }
  };

  const styling = getBoxStyling();

  const getTitle = () => {
    if (isError) return "Analysis Error";
    if (isReal) return "Why is this image authentic?";
    return "Why is this image fake?";
  };

  const getIcon = () => {
    if (isError) {
      return (
        <Error
          sx={{ fontSize: isMobile ? 28 : 32, color: styling.iconColor, mr: 2 }}
        />
      );
    } else if (isReal) {
      return (
        <CheckCircle
          sx={{ fontSize: isMobile ? 28 : 32, color: styling.iconColor, mr: 2 }}
        />
      );
    } else {
      return (
        <Psychology
          sx={{ fontSize: isMobile ? 28 : 32, color: styling.iconColor, mr: 2 }}
        />
      );
    }
  };

  return (
    <Paper
      sx={{
        p: isMobile ? 2 : 4,
        bgcolor: styling.bgcolor,
        borderRadius: 2,
        border: "1px solid",
        borderColor: styling.borderColor,
        mt: 3,
        minHeight: isMobile ? 200 : 250,
      }}
    >
      <Box sx={{ display: "flex", alignItems: "center", mb: isMobile ? 2 : 3 }}>
        {getIcon()}
        <Typography variant={isMobile ? "h6" : "h5"} color={styling.textColor}>
          {getTitle()}
        </Typography>
      </Box>
      {explanation ? (
        <Typography
          variant={isMobile ? "body2" : "body1"}
          sx={{
            lineHeight: 1.8,
            fontSize: isMobile ? "1rem" : "1.1rem",
            wordBreak: "break-word",
            overflowWrap: "break-word",
          }}
        >
          {explanation}
        </Typography>
      ) : isAnalyzing ? (
        <Typography
          variant="body2"
          color="text.secondary"
          sx={{ fontStyle: "italic" }}
        >
          {isError
            ? "Analyzing image..."
            : "Generating detailed explanation..."}
        </Typography>
      ) : null}
    </Paper>
  );
};

ExplanationBox.propTypes = {
  explanation: PropTypes.string,
  isAnalyzing: PropTypes.bool.isRequired,
  prediction: PropTypes.string,
};

export default ExplanationBox;
