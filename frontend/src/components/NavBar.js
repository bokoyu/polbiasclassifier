import { Flex, HStack, Link } from "@chakra-ui/react";
import { NavLink } from "react-router-dom";

const links = [
  { to: "/", label: "Home" },
  { to: "/evaluate", label: "Evaluate" },
  { to: "/train", label: "Train" },
];

export default function NavBar() {
  return (
    <Flex bg="white" shadow="sm" py={3} px={6}>
      <HStack spacing={6}>
        {links.map(({ to, label }) => (
          <Link
            key={to}
            as={NavLink}
            to={to}
            fontWeight="medium"
            _activeLink={{ color: "blue.500" }}
          >
            {label}
          </Link>
        ))}
      </HStack>
    </Flex>
  );
}
