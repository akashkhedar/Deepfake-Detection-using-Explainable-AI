import React, { useState } from "react";
import { Box, Typography, useTheme, useMediaQuery } from "@mui/material";
import { CloudUpload } from "@mui/icons-material";
import PropTypes from "prop-types";

const ImageUploadArea = ({ onImageSelect }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("md"));

  const handleDragEnter = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      if (file.type.startsWith("image/")) {
        // Directly call onImageSelect with the file for drag & drop
        onImageSelect(file);
      }
    }
  };

  const handleClick = () => {
    // For click, we need to trigger file picker, but since onImageSelect
    // is expecting a file directly, we need to create the file picker here
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.onchange = (e) => {
      const file = e.target.files[0];
      if (file) {
        onImageSelect(file);
      }
    };
    input.click();
  };

  return (
    <Box
      sx={{
        border: "2px dashed",
        borderColor: isDragOver ? "secondary.main" : "primary.main",
        borderRadius: 3,
        p: isMobile ? 3 : 4,
        textAlign: "center",
        bgcolor: isDragOver ? "action.selected" : "action.hover",
        transition: "all 0.3s ease",
        cursor: "pointer",
        minHeight: isMobile ? 250 : 300,
        width: "100%",
        aspectRatio: "1", // Make it square
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        "&:hover": {
          borderColor: "secondary.main",
          bgcolor: "action.selected",
        },
      }}
      onClick={handleClick}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      <CloudUpload
        sx={{
          fontSize: isMobile ? 50 : 60,
          color: isDragOver ? "secondary.main" : "primary.main",
          mb: 2,
        }}
      />
      <Typography variant={isMobile ? "body1" : "h6"} gutterBottom>
        {isDragOver ? "Drop your image here" : "Drop your image here"}
      </Typography>
      <Typography
        variant={isMobile ? "caption" : "body2"}
        color="text.secondary"
        paragraph
      >
        or click to browse files
      </Typography>
      <Typography variant="caption" color="text.secondary">
        Supports: JPG, PNG, WebP (Max: 10MB)
      </Typography>
    </Box>
  );
};

ImageUploadArea.propTypes = {
  onImageSelect: PropTypes.func.isRequired,
};

export default ImageUploadArea;
