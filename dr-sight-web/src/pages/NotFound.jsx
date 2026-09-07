import { Link } from "react-router-dom";

export default function NotFound() {
  return (
    <div className="py-32 flex flex-col items-center gap-4 text-center">
      <h1 className="font-display text-4xl font-semibold">Page not found</h1>
      <p className="text-muted-foreground">
        The page you're looking for doesn't exist in this application.
      </p>
      <Link
        to="/"
        className="text-primary text-sm font-medium hover:underline"
      >
        Back to home
      </Link>
    </div>
  );
}
