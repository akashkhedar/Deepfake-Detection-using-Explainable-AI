import GitHubIcon from "@mui/icons-material/GitHub";
import AppBar from "@mui/material/AppBar";
import Button from "@mui/material/Button";
import Toolbar from "@mui/material/Toolbar";

export default function Navbar() {
  const handleGitHub = () => {
    window.open("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", "_blank");
  };
  return (
    <AppBar
      position="static"
      elevation={0}
      sx={{
        bgcolor: "transparent",
        boxShadow: "none",
      }}
    >
      <Toolbar
        sx={{
          display: "flex",
          justifyContent: "flex-end",
        }}
      >
        <Button
          onClick={handleGitHub}
          sx={{
            color: "white",
            minWidth: "auto",
            "&:hover": {
              bgcolor: "rgba(255,255,255,0.1)",
            },
          }}
        >
          <GitHubIcon fontSize="large" />
        </Button>
      </Toolbar>
    </AppBar>
  );
}
