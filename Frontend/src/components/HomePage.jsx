import { Box, CssBaseline, ThemeProvider, createTheme } from "@mui/material";
import Navbar from "./Navbar";
import ModelSection from "./ModelSection";

// Create a dark theme
const darkTheme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: "#ffffff",
    },
    background: {
      default: "#000000",
      paper: "#121212",
    },
    text: {
      primary: "#ffffff",
    },
  },
});

const HomePage = () => {
  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <Box
        sx={{
          bgcolor: "black",
          color: "white",
          minHeight: "100vh",
          display: "flex",
          flexDirection: "column",
          width: "100%",
          overflow: "hidden",
        }}
      >
        <Box sx={{ flexShrink: 0 }}>
          <Navbar />
        </Box>
        <Box
          sx={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            overflow: "auto",
          }}
        >
          <ModelSection />
        </Box>
      </Box>
    </ThemeProvider>
  );
};

export default HomePage;
